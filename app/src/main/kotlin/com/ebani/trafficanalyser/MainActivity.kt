package com.ebani.trafficanalyser

import android.Manifest
import android.content.pm.PackageManager
import android.content.res.AssetFileDescriptor
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.MediaMetadataRetriever
import android.os.Bundle
import android.util.Size
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var localVideoView: ImageView
    private lateinit var overlayView: DetectionOverlayView
    private lateinit var statusView: TextView
    private lateinit var cameraExecutor: ExecutorService
    private var detector: VehicleDetector? = null
    private var fps = 0f
    @Volatile private var localVideoRunning = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        cameraExecutor = Executors.newSingleThreadExecutor()
        buildUi()

        try {
            detector = VehicleDetector(this).also {
                overlayView.setLabels(it.labels)
            }
            statusView.text = "VehicleNet ready"
        } catch (error: IOException) {
            statusView.text = "Model load failed"
            Toast.makeText(this, error.message, Toast.LENGTH_LONG).show()
        } catch (error: RuntimeException) {
            statusView.text = "Model load failed"
            Toast.makeText(this, error.message, Toast.LENGTH_LONG).show()
        }

        when {
            DEBUG_USE_LOCAL_VIDEO -> startLocalVideoDebug()
            hasCameraPermission() -> startCamera()
            else -> ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                CAMERA_PERMISSION_REQUEST,
            )
        }
    }

    private fun buildUi() {
        val root = FrameLayout(this)
        localVideoView = ImageView(this).apply {
            scaleType = ImageView.ScaleType.CENTER_CROP
            visibility = if (DEBUG_USE_LOCAL_VIDEO) View.VISIBLE else View.GONE
        }
        previewView = PreviewView(this).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER
            visibility = if (DEBUG_USE_LOCAL_VIDEO) View.GONE else View.VISIBLE
        }
        overlayView = DetectionOverlayView(this)
        statusView = TextView(this).apply {
            setTextColor(0xFFFFFFFF.toInt())
            textSize = 16f
            gravity = Gravity.CENTER
            setBackgroundColor(0x88000000.toInt())
            setPadding(18, 10, 18, 10)
        }

        root.addView(localVideoView, fullScreenLayoutParams())
        root.addView(previewView, fullScreenLayoutParams())
        root.addView(overlayView, fullScreenLayoutParams())

        val statusParams = FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT,
            Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL,
        ).apply {
            setMargins(16, 16, 16, 24)
        }
        root.addView(statusView, statusParams)
        setContentView(root)
    }

    private fun fullScreenLayoutParams() = FrameLayout.LayoutParams(
        ViewGroup.LayoutParams.MATCH_PARENT,
        ViewGroup.LayoutParams.MATCH_PARENT,
    )

    private fun hasCameraPermission(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray,
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (
            requestCode == CAMERA_PERMISSION_REQUEST &&
            grantResults.isNotEmpty() &&
            grantResults[0] == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            statusView.text = "Camera permission required"
        }
    }

    private fun startLocalVideoDebug() {
        val activeDetector = detector ?: return
        localVideoRunning = true
        statusView.text = "Local video debug running"
        cameraExecutor.execute {
            val retriever = MediaMetadataRetriever()
            try {
                assets.openFd(LOCAL_VIDEO_ASSET).use { descriptor ->
                    retriever.setDataSourceFrom(descriptor)
                    val durationMs = retriever
                        .extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                        ?.toLongOrNull()
                        ?: 0L
                    var positionMs = 0L

                    while (localVideoRunning) {
                        val startedAt = System.nanoTime()
                        val frame = retriever.getFrameAtTime(
                            positionMs * 1000L,
                            MediaMetadataRetriever.OPTION_CLOSEST,
                        ) ?: run {
                            positionMs = 0L
                            continue
                        }

                        val detections = activeDetector.detect(frame)
                        val instantFps = 1_000_000_000f / maxOf(1f, (System.nanoTime() - startedAt).toFloat())
                        fps = if (fps == 0f) instantFps else fps * 0.85f + instantFps * 0.15f
                        overlayView.update(detections, frame.width, frame.height, fps)
                        localVideoView.post { localVideoView.setImageBitmap(frame) }
                        statusView.post {
                            statusView.text = String.format(Locale.US, "Local debug: %d detections", detections.size)
                        }

                        positionMs += DEBUG_FRAME_STEP_MS
                        if (durationMs > 0L && positionMs >= durationMs) {
                            positionMs = 0L
                        }

                        val elapsedMs = (System.nanoTime() - startedAt) / 1_000_000L
                        try {
                            Thread.sleep(maxOf(1L, DEBUG_FRAME_STEP_MS - elapsedMs))
                        } catch (interrupted: InterruptedException) {
                            Thread.currentThread().interrupt()
                            break
                        }
                    }
                }
            } catch (error: IOException) {
                statusView.post { statusView.text = "Local video failed" }
            } catch (error: RuntimeException) {
                statusView.post { statusView.text = "Local video failed" }
            } finally {
                try {
                    retriever.release()
                } catch (_: IOException) {
                    // Nothing useful to recover during shutdown of debug playback.
                }
            }
        }
    }

    private fun AssetFileDescriptor.setDataSourceOn(retriever: MediaMetadataRetriever) {
        retriever.setDataSource(fileDescriptor, startOffset, length)
    }

    private fun MediaMetadataRetriever.setDataSourceFrom(descriptor: AssetFileDescriptor) {
        descriptor.setDataSourceOn(this)
    }

    private fun startCamera() {
        if (detector == null) return

        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener(
            {
                try {
                    val cameraProvider = providerFuture.get()
                    val preview = Preview.Builder().build().also {
                        it.surfaceProvider = previewView.surfaceProvider
                    }
                    val analysis = ImageAnalysis.Builder()
                        .setTargetResolution(Size(640, 480))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build()
                        .also { it.setAnalyzer(cameraExecutor, ::analyzeFrame) }

                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(
                        this,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview,
                        analysis,
                    )
                    statusView.text = "Realtime analysis running"
                } catch (error: Exception) {
                    statusView.text = "Camera start failed"
                    Toast.makeText(this, error.message, Toast.LENGTH_LONG).show()
                }
            },
            ContextCompat.getMainExecutor(this),
        )
    }

    private fun analyzeFrame(imageProxy: ImageProxy) {
        val activeDetector = detector ?: run {
            imageProxy.close()
            return
        }

        val startedAt = System.nanoTime()
        try {
            val bitmap = imageProxy.toBitmapFromYuv()
            val rotated = bitmap.rotate(imageProxy.imageInfo.rotationDegrees)
            val detections = activeDetector.detect(rotated)
            val instantFps = 1_000_000_000f / maxOf(1f, (System.nanoTime() - startedAt).toFloat())
            fps = if (fps == 0f) instantFps else fps * 0.85f + instantFps * 0.15f
            overlayView.update(detections, rotated.width, rotated.height, fps)
            statusView.post { statusView.text = String.format(Locale.US, "%d detections", detections.size) }
        } catch (error: RuntimeException) {
            statusView.post { statusView.text = "Analysis error" }
        } finally {
            imageProxy.close()
        }
    }

    private fun ImageProxy.toBitmapFromYuv(): Bitmap {
        val nv21 = yuv420ToNv21()
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val stream = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 82, stream)
        val bytes = stream.toByteArray()
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }

    private fun ImageProxy.yuv420ToNv21(): ByteArray {
        val nv21 = ByteArray(width * height * 3 / 2)
        val yPlane = planes[0]
        val uPlane = planes[1]
        val vPlane = planes[2]
        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer

        var outputIndex = 0
        for (row in 0 until height) {
            val yRowStart = row * yPlane.rowStride
            for (col in 0 until width) {
                nv21[outputIndex++] = yBuffer.get(yRowStart + col * yPlane.pixelStride)
            }
        }

        val chromaHeight = height / 2
        val chromaWidth = width / 2
        for (row in 0 until chromaHeight) {
            val uRowStart = row * uPlane.rowStride
            val vRowStart = row * vPlane.rowStride
            for (col in 0 until chromaWidth) {
                nv21[outputIndex++] = vBuffer.get(vRowStart + col * vPlane.pixelStride)
                nv21[outputIndex++] = uBuffer.get(uRowStart + col * uPlane.pixelStride)
            }
        }
        return nv21
    }

    private fun Bitmap.rotate(rotationDegrees: Int): Bitmap {
        if (rotationDegrees == 0) return this
        val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
        return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
    }

    override fun onDestroy() {
        super.onDestroy()
        localVideoRunning = false
        detector?.close()
        cameraExecutor.shutdown()
    }

    companion object {
        private const val CAMERA_PERMISSION_REQUEST = 7
        private const val DEBUG_USE_LOCAL_VIDEO = true
        private const val LOCAL_VIDEO_ASSET = "local_traffic.mp4"
        private const val DEBUG_FRAME_STEP_MS = 120L
    }
}
