package com.ebani.trafficanalyser

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.SurfaceTexture
import android.media.MediaPlayer
import android.os.Bundle
import android.util.Size
import android.view.Gravity
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
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
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger

/**
 * Quad-Engine Traffic Analyzer.
 * Maximum parallelization and predictive tracking for "Live Stream" grade detection.
 */
class MainActivity : AppCompatActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var localVideoTexture: TextureView
    private lateinit var overlayView: DetectionOverlayView
    private lateinit var statusView: TextView
    
    private lateinit var cameraExecutor: ExecutorService
    private var mediaPlayer: MediaPlayer? = null
    private var detector: VehicleDetector? = null
    private var fps = 0f
    
    // Quad-Buffering for parallel sampling
    private val samplingBitmaps = arrayOfNulls<Bitmap>(4)
    private val samplingIndex = AtomicInteger(0)
    
    private val activeInferences = AtomicInteger(0)
    private val maxParallelTasks = 4

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // High-priority multi-threaded AI pool
        cameraExecutor = Executors.newFixedThreadPool(maxParallelTasks) { runnable ->
            Thread(runnable).apply {
                priority = Thread.MAX_PRIORITY
                name = "AI-Quad-Engine"
            }
        }
        
        buildUi()

        try {
            detector = VehicleDetector(this).also {
                overlayView.setLabels(it.labels)
                for (i in 0 until 4) {
                    samplingBitmaps[i] = Bitmap.createBitmap(it.inputWidth, it.inputHeight, Bitmap.Config.ARGB_8888)
                }
            }
            statusView.text = "Quad-Engine System: ACTIVE"
        } catch (error: Exception) {
            statusView.text = "Init Failed"
            Toast.makeText(this, "Error: ${error.message}", Toast.LENGTH_LONG).show()
        }

        if (DEBUG_USE_LOCAL_VIDEO) {
            setupLocalVideoPlayer()
        } else if (hasCameraPermission()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), CAMERA_PERMISSION_REQUEST)
        }
    }

    private fun buildUi() {
        val root = FrameLayout(this)
        localVideoTexture = TextureView(this).apply {
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

        root.addView(localVideoTexture, fullScreenLayoutParams())
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

    private fun setupLocalVideoPlayer() {
        localVideoTexture.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
                startMediaPlayer(Surface(surface))
            }
            override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {}
            override fun onSurfaceTextureDestroyed(surface: SurfaceTexture) = true
            override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {
                sampleFrameForDetection()
            }
        }
    }

    private fun startMediaPlayer(surface: Surface) {
        try {
            mediaPlayer = MediaPlayer().apply {
                setSurface(surface)
                assets.openFd(LOCAL_VIDEO_ASSET).use { fd ->
                    setDataSource(fd.fileDescriptor, fd.startOffset, fd.length)
                }
                isLooping = true
                setOnPreparedListener { it.start() }
                prepareAsync()
            }
        } catch (e: Exception) {
            statusView.text = "Video Playback Error"
        }
    }

    private fun sampleFrameForDetection() {
        val activeDetector = detector ?: return
        if (activeInferences.get() >= maxParallelTasks) return

        activeInferences.incrementAndGet()
        val bufferIndex = samplingIndex.getAndIncrement() % 4
        val bitmapBuffer = samplingBitmaps[bufferIndex] ?: run { activeInferences.decrementAndGet(); return }
        
        localVideoTexture.getBitmap(bitmapBuffer)
        
        cameraExecutor.execute {
            try {
                val startAi = System.nanoTime()
                val detections = activeDetector.detect(bitmapBuffer)
                val elapsedAi = (System.nanoTime() - startAi).toFloat()
                
                val instantFps = 1_000_000_000f / maxOf(1f, elapsedAi)
                fps = if (fps == 0f) instantFps else fps * 0.9f + (instantFps * maxParallelTasks) * 0.1f
                
                overlayView.update(detections, bitmapBuffer.width, bitmapBuffer.height, fps)
                statusView.post {
                    statusView.text = String.format(Locale.US, "ULTRA-SYNC: %d vehicles (%.1f ms)", detections.size, elapsedAi / 1_000_000f)
                }
            } finally {
                activeInferences.decrementAndGet()
            }
        }
    }

    private fun startCamera() {
        val activeDetector = detector ?: return
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            try {
                val cameraProvider = providerFuture.get()
                val preview = Preview.Builder().build().also {
                    it.surfaceProvider = previewView.surfaceProvider
                }
                val analysis = ImageAnalysis.Builder()
                    .setTargetResolution(Size(activeDetector.inputWidth, activeDetector.inputHeight))
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also { it.setAnalyzer(cameraExecutor, ::analyzeFrame) }

                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
            } catch (error: Exception) {
                statusView.text = "Camera Failed"
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeFrame(imageProxy: ImageProxy) {
        val activeDetector = detector ?: run { imageProxy.close(); return }
        if (activeInferences.get() >= maxParallelTasks) {
            imageProxy.close()
            return
        }

        activeInferences.incrementAndGet()
        val startedAt = System.nanoTime()
        
        try {
            val buffer = imageProxy.planes[0].buffer
            val rotation = imageProxy.imageInfo.rotationDegrees
            val detections = activeDetector.detectRgba(buffer, imageProxy.width, imageProxy.height, rotation)

            val elapsed = (System.nanoTime() - startedAt).toFloat()
            val instantFps = 1_000_000_000f / maxOf(1f, elapsed)
            fps = if (fps == 0f) instantFps else fps * 0.9f + (instantFps * maxParallelTasks) * 0.1f

            val isRotated = rotation == 90 || rotation == 270
            val targetWidth = if (isRotated) imageProxy.height else imageProxy.width
            val targetHeight = if (isRotated) imageProxy.width else imageProxy.height

            overlayView.update(detections, targetWidth, targetHeight, fps)
        } finally {
            imageProxy.close()
            activeInferences.decrementAndGet()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        mediaPlayer?.release()
        samplingBitmaps.forEach { it?.recycle() }
        detector?.close()
        cameraExecutor.shutdown()
    }

    private fun hasCameraPermission(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_REQUEST && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        }
    }

    companion object {
        private const val CAMERA_PERMISSION_REQUEST = 7
        private const val DEBUG_USE_LOCAL_VIDEO = true 
        private const val LOCAL_VIDEO_ASSET = "local_traffic.mp4"
    }
}
