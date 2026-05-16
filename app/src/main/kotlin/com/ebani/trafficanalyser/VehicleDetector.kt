package com.ebani.trafficanalyser

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.Locale
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.max
import kotlin.math.min

/**
 * Ultra-Performance Quad-Engine Vehicle Detector.
 * Processes up to 4 frames in parallel for seamless, high-frequency surveillance.
 */
class VehicleDetector(context: Context) : AutoCloseable {
    
    // Quad-Core AI Engine for extreme throughput
    private val poolSize = 4
    private val interpreters = arrayOfNulls<Interpreter>(poolSize)
    private val gpuDelegates = arrayOfNulls<GpuDelegate>(poolSize)
    private val nnApiDelegates = arrayOfNulls<NnApiDelegate>(poolSize)
    
    private val inputBuffers = arrayOfNulls<ByteBuffer>(poolSize)
    private val outputBuffers = Array(poolSize) { mutableMapOf<Int, ByteBuffer>() }
    private val outputArrays = Array(poolSize) { mutableMapOf<Int, FloatArray>() }
    private val outputs = Array(poolSize) { mutableMapOf<Int, Any>() }
    
    private val nextIndex = AtomicInteger(0)
    
    val labels: List<String>
    val inputWidth: Int
    val inputHeight: Int
    private val inputType: DataType
    private val inputScale: Float
    private val inputZeroPoint: Int
    
    private var cachedPixels: IntArray? = null

    init {
        labels = loadLabels(context)
        val modelBuffer = loadModel(context, MODEL_ASSET)
        
        // Metadata extraction
        val tempInterpreter = Interpreter(modelBuffer, Interpreter.Options())
        val inputTensor = tempInterpreter.getInputTensor(0)
        val shape = inputTensor.shape()
        inputHeight = shape[1]
        inputWidth = shape[2]
        inputType = inputTensor.dataType()
        inputScale = inputTensor.quantizationParams().scale
        inputZeroPoint = inputTensor.quantizationParams().zeroPoint
        tempInterpreter.close()

        val compatList = CompatibilityList()
        val bytesPerChannel = if (inputType == DataType.FLOAT32) 4 else 1

        for (i in 0 until poolSize) {
            val options = Interpreter.Options()
            if (compatList.isDelegateSupportedOnThisDevice) {
                gpuDelegates[i] = GpuDelegate(compatList.bestOptionsForThisDevice)
                options.addDelegate(gpuDelegates[i])
            } else {
                try {
                    nnApiDelegates[i] = NnApiDelegate()
                    options.addDelegate(nnApiDelegates[i])
                } catch (e: Exception) {
                    options.setNumThreads(2)
                }
            }

            try {
                interpreters[i] = Interpreter(modelBuffer, options)
            } catch (e: Throwable) {
                interpreters[i] = Interpreter(modelBuffer, Interpreter.Options().setNumThreads(2))
            }

            val active = interpreters[i]!!
            inputBuffers[i] = ByteBuffer.allocateDirect(inputWidth * inputHeight * 3 * bytesPerChannel)
                .order(ByteOrder.nativeOrder())
                
            repeat(active.outputTensorCount) { idx ->
                val tensor = active.getOutputTensor(idx)
                val buffer = ByteBuffer.allocateDirect(tensor.numBytes()).order(ByteOrder.nativeOrder())
                outputBuffers[i][idx] = buffer
                outputArrays[i][idx] = FloatArray(tensor.numElements())
                outputs[i][idx] = buffer
            }
        }
    }

    fun detectRgba(rgbaBuffer: ByteBuffer, width: Int, height: Int, rotation: Int): List<Detection> {
        val index = nextIndex.getAndIncrement() % poolSize
        fillInputBufferFromRgba(index, rgbaBuffer, width, height, rotation)
        return runInference(index, width, height)
    }

    fun detect(bitmap: Bitmap): List<Detection> {
        val index = nextIndex.getAndIncrement() % poolSize
        val resized = if (bitmap.width == inputWidth && bitmap.height == inputHeight) {
            bitmap
        } else {
            Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true)
        }
        fillInputBufferFromBitmap(index, resized)
        return runInference(index, bitmap.width, bitmap.height)
    }

    private fun runInference(index: Int, sourceWidth: Int, sourceHeight: Int): List<Detection> {
        val interpreter = interpreters[index] ?: return emptyList()
        val input = inputBuffers[index] ?: return emptyList()
        val outMap = outputs[index]
        val buffers = outputBuffers[index]
        
        input.rewind()
        buffers.values.forEach { it.rewind() }
        interpreter.runForMultipleInputsOutputs(arrayOf(input), outMap)

        val detections = mutableListOf<Detection>()
        buffers.forEach { (idx, buffer) ->
            val tensor = interpreter.getOutputTensor(idx)
            buffer.rewind()
            detections += parseOutput(index, buffer, tensor, idx, sourceWidth, sourceHeight)
        }
        return nonMaxSuppression(detections)
    }

    private fun fillInputBufferFromRgba(index: Int, rgba: ByteBuffer, width: Int, height: Int, rotation: Int) {
        val buffer = inputBuffers[index] ?: return
        buffer.rewind()
        rgba.rewind()
        
        val isFloat = inputType == DataType.FLOAT32
        val scale = if (inputScale > 0f) inputScale else 1f
        val zeroPoint = inputZeroPoint
        
        for (y in 0 until inputHeight) {
            for (x in 0 until inputWidth) {
                val srcX: Int
                val srcY: Int
                when (rotation) {
                    90 -> { srcX = (y * width / inputHeight); srcY = height - 1 - (x * height / inputWidth) }
                    180 -> { srcX = width - 1 - (x * width / inputWidth); srcY = height - 1 - (y * height / inputHeight) }
                    270 -> { srcX = width - 1 - (y * width / inputHeight); srcY = (x * height / inputWidth) }
                    else -> { srcX = (x * width / inputWidth); srcY = (y * height / inputHeight) }
                }
                
                val pixelIndex = (srcY.coerceIn(0, height - 1) * width + srcX.coerceIn(0, width - 1)) * 4
                val r = rgba.get(pixelIndex).toInt() and 0xFF
                val g = rgba.get(pixelIndex + 1).toInt() and 0xFF
                val b = rgba.get(pixelIndex + 2).toInt() and 0xFF
                
                if (isFloat) {
                    buffer.putFloat(r / 255f); buffer.putFloat(g / 255f); buffer.putFloat(b / 255f)
                } else {
                    buffer.put(((r / scale).toInt() + zeroPoint).toByte())
                    buffer.put(((g / scale).toInt() + zeroPoint).toByte())
                    buffer.put(((b / scale).toInt() + zeroPoint).toByte())
                }
            }
        }
    }

    private fun fillInputBufferFromBitmap(index: Int, bitmap: Bitmap) {
        val buffer = inputBuffers[index] ?: return
        buffer.rewind()
        val size = inputWidth * inputHeight
        var pixels = cachedPixels ?: IntArray(size).also { cachedPixels = it }
        bitmap.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)

        val isFloat = inputType == DataType.FLOAT32
        val scale = if (inputScale > 0f) inputScale else 1f
        val zeroPoint = inputZeroPoint
        
        for (i in 0 until size) {
            val pixel = pixels[i]
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8) and 0xFF
            val b = pixel and 0xFF
            if (isFloat) {
                buffer.putFloat(r / 255f); buffer.putFloat(g / 255f); buffer.putFloat(b / 255f)
            } else {
                buffer.put(((r / scale).toInt() + zeroPoint).toByte())
                buffer.put(((g / scale).toInt() + zeroPoint).toByte())
                buffer.put(((b / scale).toInt() + zeroPoint).toByte())
            }
        }
    }

    private fun parseOutput(poolIdx: Int, buffer: ByteBuffer, tensor: Tensor, outIdx: Int, frameWidth: Int, frameHeight: Int): List<Detection> {
        val data = readOutputFloats(poolIdx, buffer, tensor, outIdx)
        val shape = tensor.shape()
        val offset = if (shape.isNotEmpty() && shape[0] == 1) 1 else 0
        val first = shape[offset]
        val second = shape[offset + 1]
        val transpose = first <= labels.size + 5 && second > first
        val rows = if (transpose) second else first
        val cols = if (transpose) first else second
        val detections = mutableListOf<Detection>()

        repeat(rows) { row ->
            val parsed = parseRow(data, row, first, second, cols, transpose) ?: return@repeat
            if (parsed.score < CONFIDENCE_THRESHOLD) return@repeat
            val box = scaleBox(parsed.x1, parsed.y1, parsed.x2, parsed.y2, frameWidth, frameHeight)
            detections += Detection(box, labels[parsed.classId].lowercase(Locale.US), parsed.score)
        }
        return detections
    }

    private fun parseRow(data: FloatArray, row: Int, first: Int, second: Int, cols: Int, transpose: Boolean): ParsedRow? {
        if (cols == 6) {
            return ParsedRow(
                data.valueAt(row, 0, second, transpose), data.valueAt(row, 1, second, transpose),
                data.valueAt(row, 2, second, transpose), data.valueAt(row, 3, second, transpose),
                data.valueAt(row, 4, second, transpose), data.valueAt(row, 5, second, transpose).toInt()
            )
        }
        if (cols < labels.size + 4) return null
        val x = data.valueAt(row, 0, second, transpose)
        val y = data.valueAt(row, 1, second, transpose)
        val w = data.valueAt(row, 2, second, transpose)
        val h = data.valueAt(row, 3, second, transpose)
        val classStart = max(4, cols - labels.size)
        val obj = if (classStart == 5) data.valueAt(row, 4, second, transpose) else 1f
        var classId = 0
        var bestScore = 0f
        for (col in classStart until cols) {
            val score = data.valueAt(row, col, second, transpose)
            if (score > bestScore) { bestScore = score; classId = col - classStart }
        }
        return ParsedRow(x - w / 2f, y - h / 2f, x + w / 2f, y + h / 2f, obj * bestScore, classId)
    }

    private fun FloatArray.valueAt(row: Int, col: Int, second: Int, transpose: Boolean): Float =
        if (transpose) this[col * second + row] else this[row * second + col]

    private fun readOutputFloats(poolIdx: Int, buffer: ByteBuffer, tensor: Tensor, outIdx: Int): FloatArray {
        val result = outputArrays[poolIdx][outIdx]!!
        val scale = tensor.quantizationParams().scale
        val zeroPoint = tensor.quantizationParams().zeroPoint
        for (i in result.indices) {
            result[i] = when (tensor.dataType()) {
                DataType.FLOAT32 -> buffer.float
                DataType.INT8 -> { val raw = buffer.get().toInt(); if (scale > 0f) (raw - zeroPoint) * scale else raw.toFloat() }
                else -> { val raw = buffer.get().toInt() and 0xFF; if (scale > 0f) (raw - zeroPoint) * scale else raw.toFloat() }
            }
        }
        return result
    }

    private fun scaleBox(x1Raw: Float, y1Raw: Float, x2Raw: Float, y2Raw: Float, frameWidth: Int, frameHeight: Int): RectF {
        var x1 = x1Raw; var y1 = y1Raw; var x2 = x2Raw; var y2 = y2Raw
        if (max(max(x1, y1), max(x2, y2)) <= 1.5f) {
            x1 *= frameWidth; x2 *= frameWidth; y1 *= frameHeight; y2 *= frameHeight
        } else {
            val sx = frameWidth / inputWidth.toFloat()
            val sy = frameHeight / inputHeight.toFloat()
            x1 *= sx; x2 *= sx; y1 *= sy; y2 *= sy
        }
        return RectF(min(x1, x2).coerceIn(0f, frameWidth - 1f), min(y1, y2).coerceIn(0f, frameHeight - 1f),
            max(x1, x2).coerceIn(0f, frameWidth - 1f), max(y1, y2).coerceIn(0f, frameHeight - 1f))
    }

    private fun nonMaxSuppression(input: List<Detection>): List<Detection> {
        val sorted = input.sortedByDescending { it.score }
        val kept = mutableListOf<Detection>()
        val removed = BooleanArray(sorted.size)
        sorted.indices.forEach { i ->
            if (removed[i]) return@forEach
            val det = sorted[i]
            kept += det
            for (j in i + 1 until sorted.size) {
                if (!removed[j] && intersectionOverUnion(det.box, sorted[j].box) > NMS_THRESHOLD) removed[j] = true
            }
        }
        return kept
    }

    private fun intersectionOverUnion(a: RectF, b: RectF): Float {
        val left = max(a.left, b.left); val top = max(a.top, b.top); val right = min(a.right, b.right); val bottom = min(a.bottom, b.bottom)
        val intersection = max(0f, right - left) * max(0f, bottom - top)
        val union = a.width() * a.height() + b.width() * b.height() - intersection
        return if (union <= 0f) 0f else intersection / union
    }

    override fun close() {
        for (i in 0 until poolSize) { interpreters[i]?.close(); gpuDelegates[i]?.close(); nnApiDelegates[i]?.close() }
    }

    private data class ParsedRow(val x1: Float, val y1: Float, val x2: Float, val y2: Float, val score: Float, val classId: Int)

    companion object {
        private const val MODEL_ASSET = "best_int8.tflite"
        private const val CONFIDENCE_THRESHOLD = 0.45f
        private const val NMS_THRESHOLD = 0.45f
        private fun loadModel(context: Context, assetName: String): MappedByteBuffer {
            val descriptor = context.assets.openFd(assetName)
            FileInputStream(descriptor.fileDescriptor).use { return it.channel.map(FileChannel.MapMode.READ_ONLY, descriptor.startOffset, descriptor.declaredLength) }
        }
        private fun loadLabels(context: Context): List<String> {
            val labels = mutableListOf<String>()
            BufferedReader(InputStreamReader(context.assets.open("vehiclenet_labels.txt"))).useLines { lines ->
                lines.filter { it.trim().isNotEmpty() }.forEach { labels += it.trim().lowercase(Locale.US) }
            }
            return labels
        }
    }
}
