package com.ebani.trafficanalyser

import android.graphics.RectF

data class Detection(
    val box: RectF,
    val label: String,
    val score: Float,
)
