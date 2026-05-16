package com.ebani.trafficanalyser

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.Locale
import kotlin.math.max
import kotlin.math.min

class VehicleDetector(context: Context) : AutoCloseable {
    private val interpreter: Interpreter
    val labels: List<String>
    private val inputTensor: Tensor
    private val inputWidth: Int
    private val inputHeight: Int
    private val inputType: DataType
    private val inputScale: Float
    private val inputZeroPoint: Int

    init {
        val options = Interpreter.Options().apply {
            setNumThreads(max(2, Runtime.getRuntime().availableProcessors() / 2))
        }
        interpreter = Interpreter(loadModel(context, MODEL_ASSET), options)
        labels = loadLabels(context)
        inputTensor = interpreter.getInputTensor(0)
        val shape = inputTensor.shape()
        inputHeight = shape[1]
        inputWidth = shape[2]
        inputType = inputTensor.dataType()
        inputScale = inputTensor.quantizationParams().scale
        inputZeroPoint = inputTensor.quantizationParams().zeroPoint
    }

    fun detect(cameraBitmap: Bitmap): List<Detection> {
        val resized = Bitmap.createScaledBitmap(cameraBitmap, inputWidth, inputHeight, true)
        val input = makeInputBuffer(resized)
        val outputs = mutableMapOf<Int, Any>()
        val outputBuffers = mutableListOf<ByteBuffer>()

        repeat(interpreter.outputTensorCount) { index ->
            val tensor = interpreter.getOutputTensor(index)
            val buffer = ByteBuffer.allocateDirect(tensor.numBytes()).order(ByteOrder.nativeOrder())
            outputBuffers += buffer
            outputs[index] = buffer
        }

        interpreter.runForMultipleInputsOutputs(arrayOf(input), outputs)

        val detections = mutableListOf<Detection>()
        outputBuffers.forEachIndexed { index, buffer ->
            val tensor = interpreter.getOutputTensor(index)
            buffer.rewind()
            detections += parseOutput(buffer, tensor, cameraBitmap.width, cameraBitmap.height)
        }
        return nonMaxSuppression(detections)
    }

    private fun makeInputBuffer(bitmap: Bitmap): ByteBuffer {
        val bytesPerChannel = if (inputType == DataType.FLOAT32) 4 else 1
        val buffer = ByteBuffer
            .allocateDirect(inputWidth * inputHeight * 3 * bytesPerChannel)
            .order(ByteOrder.nativeOrder())
        val pixels = IntArray(inputWidth * inputHeight)
        bitmap.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)

        pixels.forEach { pixel ->
            buffer.putPixelChannel((pixel shr 16) and 0xFF)
            buffer.putPixelChannel((pixel shr 8) and 0xFF)
            buffer.putPixelChannel(pixel and 0xFF)
        }
        buffer.rewind()
        return buffer
    }

    private fun ByteBuffer.putPixelChannel(value: Int) {
        when (inputType) {
            DataType.FLOAT32 -> putFloat(value / 255f)
            DataType.INT8 -> {
                val quantized = (value / max(inputScale, 1e-8f)).toInt() + inputZeroPoint
                put(quantized.coerceIn(Byte.MIN_VALUE.toInt(), Byte.MAX_VALUE.toInt()).toByte())
            }
            else -> {
                val quantized = if (inputScale > 0f) (value / inputScale).toInt() + inputZeroPoint else value
                put(quantized.coerceIn(0, 255).toByte())
            }
        }
    }

    private fun parseOutput(buffer: ByteBuffer, tensor: Tensor, frameWidth: Int, frameHeight: Int): List<Detection> {
        val data = readOutputFloats(buffer, tensor)
        val shape = tensor.shape()
        val offset = if (shape.isNotEmpty() && shape[0] == 1) 1 else 0
        val dims = shape.size - offset
        if (dims != 2 || data.isEmpty()) return emptyList()

        val first = shape[offset]
        val second = shape[offset + 1]
        val transpose = first <= labels.size + 5 && second > first
        val rows = if (transpose) second else first
        val cols = if (transpose) first else second
        val detections = mutableListOf<Detection>()

        repeat(rows) { row ->
            val parsed = parseRow(data, row, first, second, cols, transpose) ?: return@repeat
            if (parsed.score < CONFIDENCE_THRESHOLD || parsed.classId !in labels.indices) return@repeat

            val box = scaleBox(parsed.x1, parsed.y1, parsed.x2, parsed.y2, frameWidth, frameHeight)
            val areaRatio = box.width() * box.height() / max(1f, frameWidth * frameHeight.toFloat())
            if (areaRatio !in MIN_BOX_AREA_RATIO..MAX_BOX_AREA_RATIO) return@repeat

            detections += Detection(box, labels[parsed.classId].canonical(), parsed.score)
        }
        return detections
    }

    private fun parseRow(
        data: FloatArray,
        row: Int,
        first: Int,
        second: Int,
        cols: Int,
        transpose: Boolean,
    ): ParsedRow? {
        if (cols == 6) {
            return ParsedRow(
                x1 = data.valueAt(row, 0, second, transpose),
                y1 = data.valueAt(row, 1, second, transpose),
                x2 = data.valueAt(row, 2, second, transpose),
                y2 = data.valueAt(row, 3, second, transpose),
                score = data.valueAt(row, 4, second, transpose),
                classId = data.valueAt(row, 5, second, transpose).toInt(),
            )
        }
        if (cols < labels.size + 4) return null

        val x = data.valueAt(row, 0, second, transpose)
        val y = data.valueAt(row, 1, second, transpose)
        val w = data.valueAt(row, 2, second, transpose)
        val h = data.valueAt(row, 3, second, transpose)
        val classStart = max(4, cols - labels.size)
        val objectness = if (classStart == 5) data.valueAt(row, 4, second, transpose) else 1f

        var classId = 0
        var bestClassScore = 0f
        for (col in classStart until cols) {
            val classScore = data.valueAt(row, col, second, transpose)
            if (classScore > bestClassScore) {
                bestClassScore = classScore
                classId = col - classStart
            }
        }
        return ParsedRow(
            x1 = x - w / 2f,
            y1 = y - h / 2f,
            x2 = x + w / 2f,
            y2 = y + h / 2f,
            score = objectness * bestClassScore,
            classId = classId,
        )
    }

    private fun FloatArray.valueAt(row: Int, col: Int, second: Int, transpose: Boolean): Float =
        if (transpose) this[col * second + row] else this[row * second + col]

    private fun readOutputFloats(buffer: ByteBuffer, tensor: Tensor): FloatArray {
        val result = FloatArray(tensor.numElements())
        val scale = tensor.quantizationParams().scale
        val zeroPoint = tensor.quantizationParams().zeroPoint

        for (index in result.indices) {
            result[index] = when (tensor.dataType()) {
                DataType.FLOAT32 -> buffer.float
                DataType.INT8 -> {
                    val raw = buffer.get().toInt()
                    if (scale > 0f) (raw - zeroPoint) * scale else raw.toFloat()
                }
                else -> {
                    val raw = buffer.get().toInt() and 0xFF
                    if (scale > 0f) (raw - zeroPoint) * scale else raw.toFloat()
                }
            }
        }
        return result
    }

    private fun scaleBox(x1Raw: Float, y1Raw: Float, x2Raw: Float, y2Raw: Float, frameWidth: Int, frameHeight: Int): RectF {
        var x1 = x1Raw
        var y1 = y1Raw
        var x2 = x2Raw
        var y2 = y2Raw
        val maxCoordinate = max(max(x1, y1), max(x2, y2))

        if (maxCoordinate <= 1.5f) {
            x1 *= frameWidth
            x2 *= frameWidth
            y1 *= frameHeight
            y2 *= frameHeight
        } else {
            x1 *= frameWidth / inputWidth.toFloat()
            x2 *= frameWidth / inputWidth.toFloat()
            y1 *= frameHeight / inputHeight.toFloat()
            y2 *= frameHeight / inputHeight.toFloat()
        }

        return RectF(
            min(x1, x2).coerceIn(0f, frameWidth - 1f),
            min(y1, y2).coerceIn(0f, frameHeight - 1f),
            max(x1, x2).coerceIn(0f, frameWidth - 1f),
            max(y1, y2).coerceIn(0f, frameHeight - 1f),
        )
    }

    private fun nonMaxSuppression(input: List<Detection>): List<Detection> {
        val sorted = input.sortedByDescending { it.score }
        val kept = mutableListOf<Detection>()
        val removed = BooleanArray(sorted.size)

        sorted.indices.forEach { i ->
            if (removed[i]) return@forEach
            val detection = sorted[i]
            kept += detection
            for (j in i + 1 until sorted.size) {
                if (!removed[j] && intersectionOverUnion(detection.box, sorted[j].box) > NMS_THRESHOLD) {
                    removed[j] = true
                }
            }
        }
        return kept
    }

    private fun intersectionOverUnion(a: RectF, b: RectF): Float {
        val left = max(a.left, b.left)
        val top = max(a.top, b.top)
        val right = min(a.right, b.right)
        val bottom = min(a.bottom, b.bottom)
        val intersection = max(0f, right - left) * max(0f, bottom - top)
        val union = a.width() * a.height() + b.width() * b.height() - intersection
        return if (union <= 0f) 0f else intersection / union
    }

    private fun String.canonical() = lowercase(Locale.US)

    override fun close() {
        interpreter.close()
    }

    private data class ParsedRow(
        val x1: Float,
        val y1: Float,
        val x2: Float,
        val y2: Float,
        val score: Float,
        val classId: Int,
    )

    companion object {
        private const val MODEL_ASSET = "best_int8.tflite"
        private const val CONFIDENCE_THRESHOLD = 0.45f
        private const val NMS_THRESHOLD = 0.45f
        private const val MIN_BOX_AREA_RATIO = 0.002f
        private const val MAX_BOX_AREA_RATIO = 0.75f

        private fun loadModel(context: Context, assetName: String): MappedByteBuffer {
            val descriptor = context.assets.openFd(assetName)
            FileInputStream(descriptor.fileDescriptor).use { inputStream ->
                val channel = inputStream.channel
                return channel.map(FileChannel.MapMode.READ_ONLY, descriptor.startOffset, descriptor.declaredLength)
            }
        }

        private fun loadLabels(context: Context): List<String> {
            val aliases = mapOf(
                "mini-bus" to "mini-bus",
                "minibus" to "mini-bus",
                "tempo traveller" to "tempo-traveller",
                "tempo-traveller" to "tempo-traveller",
                "two wheeler" to "two-wheeler",
                "two-wheeler" to "two-wheeler",
                "three wheeler" to "three-wheeler",
                "three-wheeler" to "three-wheeler",
            )
            val labels = mutableListOf<String>()
            BufferedReader(InputStreamReader(context.assets.open("vehiclenet_labels.txt"))).useLines { lines ->
                lines.map { it.trim().lowercase(Locale.US) }
                    .filter { it.isNotEmpty() }
                    .forEach { labels += aliases[it] ?: it }
            }
            return labels
        }
    }
}
