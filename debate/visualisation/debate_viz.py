# debate/visualisation/debate_viz.py
"""
Enhanced HTML chat-style visualization for debate transcripts with interactive labeling
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import hashlib


def score_to_color(score: float, threshold: float = 0.0) -> str:
    """Convert score [-1, 1] to color based on threshold
    Above threshold = red (deceptive)
    Below threshold = green (honest)
    Opacity based on distance from threshold
    """
    distance = abs(score - threshold)
    max_opacity = 0.7
    min_opacity = 0.1
    
    # Scale opacity based on distance from threshold (closer = more transparent)
    # Max distance is 2 (from -1 to +1), so normalize by dividing by 2
    opacity = min_opacity + (max_opacity - min_opacity) * min(distance / 2.0, 1.0)
    
    if score >= threshold:
        # Red for deceptive
        return f"rgba(220, 53, 69, {opacity:.3f})"
    else:
        # Green for honest
        return f"rgba(40, 167, 69, {opacity:.3f})"


def clean_tokens(tokens: List[str]) -> List[str]:
    """Clean tokenizer artifacts"""
    return [token.replace('Ġ', ' ').replace('Ċ', '\n') for token in tokens]







def create_debate_chat_visualization(transcript: Dict, output_path: Path):
    """Create enhanced interactive HTML visualization for a debate transcript"""
    
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Interactive Debate Analysis: {{ debate_id }}</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f0f2f5;
            color: #1c1e21;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 1fr 350px;
            gap: 20px;
        }
        
        .debate-section {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .controls {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            position: sticky;
            top: 20px;
            height: fit-content;
        }
        
        .context-box {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 20px;
            max-height: 300px;
            overflow-y: auto;
        }
        
        .statement {
            padding: 12px;
            margin: 10px 0;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.3s;
            border: 2px solid transparent;
        }
        
        .statement:hover {
            border-color: #667eea;
        }
        
        .statement.selected-deceptive {
            border-color: #dc3545;
            background-color: rgba(220, 53, 69, 0.05);
        }
        
        .statement.selected-honest {
            border-color: #28a745;
            background-color: rgba(40, 167, 69, 0.05);
        }
        
        .token {
            display: inline-block;
            padding: 2px 4px;
            margin: 1px;
            border-radius: 3px;
            font-family: monospace;
            font-size: 14px;
            transition: all 0.3s;
        }
        
        .slider-container {
            margin: 20px 0;
        }
        
        .slider {
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: linear-gradient(to right, #28a745 0%, #ffc107 50%, #dc3545 100%);
            outline: none;
            -webkit-appearance: none;
        }
        
        .slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #667eea;
            cursor: pointer;
        }
        
        .metrics {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            margin: 15px 0;
        }
        
        .metrics-section {
            margin-bottom: 15px;
        }
        
        .metrics-section h4 {
            margin-bottom: 8px;
            color: #667eea;
        }
        
        .plots-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 30px;
        }
        
        .plot {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        canvas {
            max-width: 100%;
            height: auto;
        }
        
        .export-btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            margin-top: 15px;
            width: 100%;
        }
        
        .export-btn:hover {
            background: #5a67d8;
        }
        
        .speaker-label {
            font-weight: bold;
            margin-bottom: 5px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .role-badge {
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: normal;
        }
        
        .role-badge.honest {
            background: #d4edda;
            color: #155724;
        }
        
        .role-badge.dishonest {
            background: #f8d7da;
            color: #721c24;
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@2.1.0"></script>
</head>
<body>
    <div class="header">
        <h1>🔬 Interactive Debate Analysis</h1>
        <div>{{ debate_id }}</div>
    </div>
    
    <div class="container">
        <div class="main-content">
            <div class="debate-section">
                <!-- Context -->
                <div class="context-box">
                    <h3>📚 Context</h3>
                    <div id="context-content">
                        {{ context_html | safe }}
                    </div>
                </div>
                
                <!-- Debate Transcript -->
                <h3>💬 Debate Transcript</h3>
                <div id="transcript">
                    {% for turn in turns %}
                    {% set turn_index = loop.index0 %}
                    <div class="turn" data-turn-id="{{ turn_index }}">
                        <div class="speaker-label">
                            {{ turn.speaker_label }}
                            <span class="role-badge {{ turn.role }}">{{ turn.role.upper() }}</span>
                            - Round {{ turn.round }}
                        </div>
                        
                        {% for statement in turn.statements %}
                        <div class="statement" 
                             data-turn="{{ turn_index }}"
                             data-statement="{{ statement.index }}"
                             data-role="{{ turn.role }}"
                             data-score="{{ statement.mean_score }}"
                             onclick="toggleLabel(this)">
                            <div class="statement-tokens">
                                {% for token, score in statement.token_scores %}
                                <span class="token" 
                                      data-token-index="{{ loop.index0 }}"
                                      data-score="{{ score }}"
                                      style="background-color: {{ score_to_color(score, 0.0) }}">{{ token }}</span>
                                {% endfor %}
                            </div>
                            <div class="statement-meta" style="font-size: 12px; color: #666; margin-top: 5px;">
                                Score: {{ '%.3f' | format(statement.mean_score) }}
                            </div>
                        </div>
                        {% endfor %}
                    </div>
                    {% endfor %}
                </div>
            </div>
            
            <div class="controls">
                <h3>🎚️ Controls</h3>
                
                <div class="slider-container">
                    <label>Threshold: <span id="threshold-value">0.000</span></label>
                    <input type="range" class="slider" id="threshold-slider" 
                           min="-1" max="1" step="0.001" value="0">
                </div>

                <button class="export-btn" id="flip-btn" onclick="flipProbeClasses()" 
                        style="margin-top: 10px; background: #e74c3c;">
                    🔄 Flip Probe Classes
                </button>
                
                <div class="metrics">
                    <h4>📊 Metrics</h4>
                    
                    <div class="metrics-section">
                        <h4>Token-Level</h4>
                        <div>AUROC: <span id="token-auroc-value">-</span></div>
                        <div>Total Tokens: <span id="token-count">-</span></div>
                        <div>Above Threshold: <span id="token-above-count">-</span></div>
                    </div>
                    
                    <div class="metrics-section">
                        <h4>Statement-Level</h4>
                        <div>AUROC: <span id="statement-auroc-value">-</span></div>
                        <div>Accuracy: <span id="accuracy-value">-</span></div>
                        <div>Total Statements: <span id="statement-count">-</span></div>
                        <div>Above Threshold: <span id="statement-above-count">-</span></div>
                    </div>
                </div>
                
                <button class="export-btn" onclick="exportLabels()">
                    💾 Export Corrected Labels
                </button>
                
                <div style="margin-top: 15px; font-size: 12px; color: #666;">
                    <p>🖱️ Click statements to toggle label</p>
                    <p>🟢 Green = Honest (score < threshold)</p>
                    <p>🔴 Red = Deceptive (score ≥ threshold)</p>
                </div>
            </div>
        </div>
        
        <div class="plots-container">
            <div class="plot">
                <h4>Token-Level Distribution</h4>
                <canvas id="token-dist-chart"></canvas>
            </div>
            <div class="plot">
                <h4>Statement-Level Distribution</h4>
                <canvas id="statement-dist-chart"></canvas>
            </div>
            <div class="plot">
                <h4>ROC Curve (Statement-Level)</h4>
                <canvas id="roc-chart"></canvas>
            </div>
            <div class="plot">
                <h4>Score Statistics</h4>
                <div id="stats-display" style="padding: 10px;">
                    <!-- Will be populated by JS -->
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // ============================================================================
        // GLOBAL STATE
        // ============================================================================
        let statementLabels = {};  // Maps statement ID to label (0=honest, 1=deceptive)
        let currentThreshold = 0.0;
        let tokenDistChart, statementDistChart, rocChart;
        let isFlipped = false;
        
        // Separate tracking for tokens and statements
        let tokenData = {
            scores: [],
            labels: [],
            statementKeys: []  // Which statement each token belongs to
        };
        
        let statementData = {
            scores: [],
            labels: [],
            keys: []
        };

        // ============================================================================
        // INITIALIZATION (runs when DOM is ready)
        // ============================================================================
        window.addEventListener('DOMContentLoaded', () => {
            console.log("[INIT] Starting initialization...");
            
            // Initialize statement labels and collect all data
            let statementCount = 0;
            let tokenCount = 0;
            
            document.querySelectorAll('.statement').forEach(stmt => {
                const key = `${stmt.dataset.turn}-${stmt.dataset.statement}`;
                const role = stmt.dataset.role;
                const meanScore = parseFloat(stmt.dataset.score);
                
                // Initialize label: dishonest = deceptive (1), honest = honest (0)
                statementLabels[key] = role === 'dishonest' ? 1 : 0;
                statementCount++;
                
                // Store statement-level data
                statementData.keys.push(key);
                statementData.scores.push(meanScore);
                statementData.labels.push(statementLabels[key]);
                
                // Collect token-level data
                stmt.querySelectorAll('.token').forEach(token => {
                    const tokenScore = parseFloat(token.dataset.score);
                    tokenData.scores.push(tokenScore);
                    tokenData.labels.push(statementLabels[key]);  // Inherit from statement
                    tokenData.statementKeys.push(key);
                    tokenCount++;
                });
                
                updateStatementStyle(stmt);
            });
            
            console.log(`[INIT] Found ${statementCount} statements with ${tokenCount} total tokens`);
            console.log(`[INIT] Statement data:`, statementData.keys.length, statementData.scores.length, statementData.labels.length);
            console.log(`[INIT] Token data:`, tokenData.scores.length, tokenData.labels.length);
            
            // Initialize charts
            initializeCharts();
            
            // Initial metrics and plots
            updateMetrics();
            updatePlots();
            
            // Threshold slider listener
            document.getElementById('threshold-slider').addEventListener('input', (e) => {
                currentThreshold = parseFloat(e.target.value);
                document.getElementById('threshold-value').textContent = currentThreshold.toFixed(3);
                updateColors();
                updateMetrics();
                updatePlots();
            });
            
            console.log("[INIT] Initialization complete");
        });

        // ============================================================================
        // HELPER FUNCTIONS
        // ============================================================================
        
        function getEffectiveScore(score) {
            return isFlipped ? (-score) : score;
        }
        
        function flipProbeClasses() {
            isFlipped = !isFlipped;
            const btn = document.getElementById('flip-btn');
            btn.textContent = isFlipped ? '🔄 Un-flip Probe Classes' : '🔄 Flip Probe Classes';
            btn.style.background = isFlipped ? '#2ecc71' : '#e74c3c';
            
            updateColors();
            updateMetrics();
            updatePlots();
        }
        
        function toggleLabel(element) {
            const key = `${element.dataset.turn}-${element.dataset.statement}`;
            statementLabels[key] = 1 - statementLabels[key];  // Toggle between 0 and 1
            
            // Update statement data
            const stmtIndex = statementData.keys.indexOf(key);
            if (stmtIndex !== -1) {
                statementData.labels[stmtIndex] = statementLabels[key];
            }
            
            // Update all token labels for this statement
            tokenData.labels = tokenData.labels.map((label, i) => 
                tokenData.statementKeys[i] === key ? statementLabels[key] : label
            );
            
            updateStatementStyle(element);
            updateMetrics();
            updatePlots();
        }
        
        function updateStatementStyle(element) {
            const key = `${element.dataset.turn}-${element.dataset.statement}`;
            const label = statementLabels[key];
            
            element.classList.remove('selected-honest', 'selected-deceptive');
            if (label === 1) {
                element.classList.add('selected-deceptive');
            } else {
                element.classList.add('selected-honest');
            }
        }

        function updateColors() {
            document.querySelectorAll('.token').forEach(token => {
                const rawScore = parseFloat(token.dataset.score);
                const score = getEffectiveScore(rawScore);
                const distance = Math.abs(score - currentThreshold);
                const maxOpacity = 0.7;
                const minOpacity = 0.1;
                
                // Max distance is 2 (from -1 to +1)
                const opacity = minOpacity + (maxOpacity - minOpacity) * Math.min(distance / 2.0, 1.0);
                
                if (score >= currentThreshold) {
                    token.style.backgroundColor = `rgba(220, 53, 69, ${opacity})`;
                } else {
                    token.style.backgroundColor = `rgba(40, 167, 69, ${opacity})`;
                }
            });
        }

        // ============================================================================
        // METRICS CALCULATION
        // ============================================================================
        
        function calculateAUROC(labels, scores) {
            // Handle edge cases
            if (labels.length === 0 || scores.length === 0) {
                return 0.5;
            }
            
            const uniqueLabels = [...new Set(labels)];
            if (uniqueLabels.length < 2) {
                return 0.5;  // Need both classes for AUROC
            }
            
            // Create pairs and sort by score
            const pairs = labels.map((label, i) => ({label, score: scores[i]}));
            pairs.sort((a, b) => b.score - a.score);  // Sort descending by score
            
            const n_pos = labels.filter(l => l === 1).length;
            const n_neg = labels.filter(l => l === 0).length;
            
            if (n_pos === 0 || n_neg === 0) {
                return 0.5;
            }
            
            // Calculate AUC using trapezoidal rule
            let auc = 0;
            let tp = 0, fp = 0;
            let tp_prev = 0, fp_prev = 0;
            
            for (let i = 0; i < pairs.length; i++) {
                if (pairs[i].label === 1) {
                    tp++;
                } else {
                    fp++;
                }
                
                // When score changes (or at last element), add trapezoid
                if (i === pairs.length - 1 || pairs[i].score !== pairs[i + 1].score) {
                    const tpr = tp / n_pos;
                    const fpr = fp / n_neg;
                    const tpr_prev_val = tp_prev / n_pos;
                    const fpr_prev_val = fp_prev / n_neg;
                    
                    // Trapezoidal area
                    auc += (fpr - fpr_prev_val) * (tpr + tpr_prev_val) / 2;
                    
                    tp_prev = tp;
                    fp_prev = fp;
                }
            }
            
            return auc;
        }
        
        function getROCPoints(labels, scores) {
            // Handle edge cases
            if (labels.length === 0 || scores.length === 0) {
                return [{x: 0, y: 0}, {x: 1, y: 1}];
            }
            
            const uniqueLabels = [...new Set(labels)];
            if (uniqueLabels.length < 2) {
                return [{x: 0, y: 0}, {x: 1, y: 1}];
            }
            
            // Get unique thresholds
            const thresholds = [...new Set(scores)].sort((a, b) => b - a);
            const points = [];
            
            const n_pos = labels.filter(l => l === 1).length;
            const n_neg = labels.filter(l => l === 0).length;
            
            if (n_pos === 0 || n_neg === 0) {
                return [{x: 0, y: 0}, {x: 1, y: 1}];
            }
            
            // Add point at (0, 0) for threshold = +infinity
            points.push({x: 0, y: 0, threshold: Infinity});
            
            // Calculate ROC points for each threshold
            thresholds.forEach(thresh => {
                const predictions = scores.map(s => s >= thresh ? 1 : 0);
                const tp = predictions.filter((p, i) => p === 1 && labels[i] === 1).length;
                const fp = predictions.filter((p, i) => p === 1 && labels[i] === 0).length;
                
                const tpr = tp / n_pos;
                const fpr = fp / n_neg;
                
                points.push({x: fpr, y: tpr, threshold: thresh});
            });
            
            // Add point at (1, 1) for threshold = -infinity
            points.push({x: 1, y: 1, threshold: -Infinity});
            
            return points;
        }
        
        function updateMetrics() {
            // Get effective scores
            const effectiveTokenScores = tokenData.scores.map(s => getEffectiveScore(s));
            const effectiveStatementScores = statementData.scores.map(s => getEffectiveScore(s));
            
            // Token-level metrics
            const tokenAUROC = calculateAUROC(tokenData.labels, effectiveTokenScores);
            const tokensAbove = effectiveTokenScores.filter(s => s >= currentThreshold).length;
            
            document.getElementById('token-auroc-value').textContent = tokenAUROC.toFixed(3);
            document.getElementById('token-count').textContent = tokenData.scores.length;
            document.getElementById('token-above-count').textContent = `${tokensAbove}/${tokenData.scores.length}`;
            
            // Statement-level metrics
            const statementAUROC = calculateAUROC(statementData.labels, effectiveStatementScores);
            const statementPredictions = effectiveStatementScores.map(s => s >= currentThreshold ? 1 : 0);
            const correct = statementPredictions.filter((p, i) => p === statementData.labels[i]).length;
            const accuracy = statementData.labels.length > 0 ? correct / statementData.labels.length : 0;
            const statementsAbove = effectiveStatementScores.filter(s => s >= currentThreshold).length;
            
            document.getElementById('statement-auroc-value').textContent = statementAUROC.toFixed(3);
            document.getElementById('accuracy-value').textContent = accuracy.toFixed(3);
            document.getElementById('statement-count').textContent = statementData.scores.length;
            document.getElementById('statement-above-count').textContent = `${statementsAbove}/${statementData.scores.length}`;
        }

        // ============================================================================
        // CHART INITIALIZATION
        // ============================================================================
        
        function initializeCharts() {
            // Token distribution chart
            const tokenCtx = document.getElementById('token-dist-chart').getContext('2d');
            tokenDistChart = new Chart(tokenCtx, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Honest',
                        backgroundColor: 'rgba(40, 167, 69, 0.5)',
                        data: []
                    }, {
                        label: 'Deceptive',
                        backgroundColor: 'rgba(220, 53, 69, 0.5)',
                        data: []
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Count'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Score'
                            }
                        }
                    },
                    plugins: {
                        annotation: {
                            annotations: {
                                threshold: {
                                    type: 'line',
                                    xMin: 0,
                                    xMax: 0,
                                    borderColor: 'black',
                                    borderWidth: 2,
                                    borderDash: [5, 5],
                                    label: {
                                        content: 'Threshold',
                                        enabled: true
                                    }
                                }
                            }
                        }
                    }
                }
            });
            
            // Statement distribution chart
            const stmtCtx = document.getElementById('statement-dist-chart').getContext('2d');
            statementDistChart = new Chart(stmtCtx, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Honest',
                        backgroundColor: 'rgba(40, 167, 69, 0.5)',
                        data: []
                    }, {
                        label: 'Deceptive',
                        backgroundColor: 'rgba(220, 53, 69, 0.5)',
                        data: []
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Count'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Score'
                            }
                        }
                    },
                    plugins: {
                        annotation: {
                            annotations: {
                                threshold: {
                                    type: 'line',
                                    xMin: 0,
                                    xMax: 0,
                                    borderColor: 'black',
                                    borderWidth: 2,
                                    borderDash: [5, 5],
                                    label: {
                                        content: 'Threshold',
                                        enabled: true
                                    }
                                }
                            }
                        }
                    }
                }
            });
            
            // ROC chart
            const rocCtx = document.getElementById('roc-chart').getContext('2d');
            rocChart = new Chart(rocCtx, {
                type: 'line',
                data: {
                    datasets: [{
                        label: 'ROC',
                        borderColor: 'rgb(75, 192, 192)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        data: [],
                        tension: 0,
                        fill: false
                    }, {
                        label: 'Current Threshold',
                        borderColor: 'rgb(255, 99, 132)',
                        backgroundColor: 'rgb(255, 99, 132)',
                        data: [],
                        pointRadius: 8,
                        pointHoverRadius: 10,
                        showLine: false
                    }]
                },
                options: {
                    responsive: true,
                    aspectRatio: 1,
                    scales: {
                        x: {
                            type: 'linear',
                            position: 'bottom',
                            title: {
                                display: true,
                                text: 'False Positive Rate'
                            },
                            min: 0,
                            max: 1
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'True Positive Rate'
                            },
                            min: 0,
                            max: 1
                        }
                    }
                }
            });
        }

        // ============================================================================
        // CHART UPDATES
        // ============================================================================
        
        function updatePlots() {
            const effectiveTokenScores = tokenData.scores.map(s => getEffectiveScore(s));
            const effectiveStatementScores = statementData.scores.map(s => getEffectiveScore(s));
            
            // Update token distribution
            updateDistributionChart(tokenDistChart, tokenData.labels, effectiveTokenScores, 'Token');
            
            // Update statement distribution
            updateDistributionChart(statementDistChart, statementData.labels, effectiveStatementScores, 'Statement');
            
            // Update ROC curve
            updateROCChart(statementData.labels, effectiveStatementScores);
            
            // Update statistics
            updateStatistics(effectiveTokenScores, tokenData.labels, effectiveStatementScores, statementData.labels);
        }
        
        function updateDistributionChart(chart, labels, scores, type) {
            const honest = scores.filter((s, i) => labels[i] === 0);
            const deceptive = scores.filter((s, i) => labels[i] === 1);
            
            const bins = 20;
            const min = Math.min(...scores);
            const max = Math.max(...scores);
            const binWidth = (max - min) / bins;
            
            const honestHist = new Array(bins).fill(0);
            const deceptiveHist = new Array(bins).fill(0);
            
            honest.forEach(s => {
                const bin = Math.min(Math.floor((s - min) / binWidth), bins - 1);
                honestHist[bin]++;
            });
            
            deceptive.forEach(s => {
                const bin = Math.min(Math.floor((s - min) / binWidth), bins - 1);
                deceptiveHist[bin]++;
            });
            
            const binLabels = Array.from({length: bins}, (_, i) => 
                (min + i * binWidth).toFixed(2));
            
            chart.data.labels = binLabels;
            chart.data.datasets[0].data = honestHist;
            chart.data.datasets[1].data = deceptiveHist;
            
            // Update threshold line
            const thresholdBin = Math.floor((currentThreshold - min) / binWidth);
            const thresholdX = binLabels[Math.max(0, Math.min(thresholdBin, bins - 1))];
            
            if (chart.options.plugins && chart.options.plugins.annotation) {
                chart.options.plugins.annotation.annotations.threshold.xMin = thresholdX;
                chart.options.plugins.annotation.annotations.threshold.xMax = thresholdX;
            }
            
            chart.update();
        }
        
        function updateROCChart(labels, scores) {
            const points = getROCPoints(labels, scores);
            const auroc = calculateAUROC(labels, scores);
            
            // Find current threshold point
            let currentPoint = {x: 0, y: 0};
            const predictions = scores.map(s => s >= currentThreshold ? 1 : 0);
            const tp = predictions.filter((p, i) => p === 1 && labels[i] === 1).length;
            const fp = predictions.filter((p, i) => p === 1 && labels[i] === 0).length;
            const n_pos = labels.filter(l => l === 1).length;
            const n_neg = labels.filter(l => l === 0).length;
            
            if (n_pos > 0 && n_neg > 0) {
                currentPoint = {
                    x: fp / n_neg,
                    y: tp / n_pos
                };
            }
            
            rocChart.data.datasets[0].data = points;
            rocChart.data.datasets[1].data = [currentPoint];
            rocChart.data.datasets[0].label = `ROC (AUC = ${auroc.toFixed(3)})`;
            rocChart.update();
        }
                
        function updateStatistics(tokenScores, tokenLabels, stmtScores, stmtLabels) {
            const tokenHonest = tokenScores.filter((s, i) => tokenLabels[i] === 0);
            const tokenDeceptive = tokenScores.filter((s, i) => tokenLabels[i] === 1);
            const stmtHonest = stmtScores.filter((s, i) => stmtLabels[i] === 0);
            const stmtDeceptive = stmtScores.filter((s, i) => stmtLabels[i] === 1);
            
            const html = `
                <div style="margin-bottom: 15px;">
                    <strong>Token-Level</strong><br>
                    Honest: μ=${mean(tokenHonest).toFixed(3)}, σ=${std(tokenHonest).toFixed(3)}<br>
                    Deceptive: μ=${mean(tokenDeceptive).toFixed(3)}, σ=${std(tokenDeceptive).toFixed(3)}<br>
                    Separation: ${(mean(tokenDeceptive) - mean(tokenHonest)).toFixed(3)}<br>
                    Count: ${tokenScores.length} tokens
                </div>
                <div>
                    <strong>Statement-Level</strong><br>
                    Honest: μ=${mean(stmtHonest).toFixed(3)}, σ=${std(stmtHonest).toFixed(3)}<br>
                    Deceptive: μ=${mean(stmtDeceptive).toFixed(3)}, σ=${std(stmtDeceptive).toFixed(3)}<br>
                    Separation: ${(mean(stmtDeceptive) - mean(stmtHonest)).toFixed(3)}<br>
                    Count: ${stmtScores.length} statements
                </div>
            `;
            
            document.getElementById('stats-display').innerHTML = html;
        }
        
        function mean(arr) {
            return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
        }
        
        function std(arr) {
            if (!arr.length) return 0;
            const m = mean(arr);
            return Math.sqrt(arr.reduce((sq, n) => sq + Math.pow(n - m, 2), 0) / arr.length);
        }
        
        function exportLabels() {
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            
            // Collect token-level data for each statement
            const tokenLevelData = [];
            document.querySelectorAll('.statement').forEach(stmt => {
                const key = `${stmt.dataset.turn}-${stmt.dataset.statement}`;
                const tokens = [];
                stmt.querySelectorAll('.token').forEach(token => {
                    tokens.push({
                        text: token.textContent,
                        score: parseFloat(token.dataset.score),
                        index: parseInt(token.dataset.tokenIndex)
                    });
                });
                
                tokenLevelData.push({
                    key: key,
                    turn: parseInt(stmt.dataset.turn),
                    statement_index: parseInt(stmt.dataset.statement),
                    label: statementLabels[key],
                    tokens: tokens
                });
            });
            
            const exportData = {
                debate_id: '{{ debate_id }}',
                timestamp: timestamp,
                threshold: currentThreshold,
                isFlipped: isFlipped,
                
                // Statement-level labels (for probe_analysis.py compatibility)
                statementLabels: statementLabels,
                
                // Token-level data
                tokenLevelData: tokenLevelData,
                
                // Statement scores for reference
                statementScores: Object.fromEntries(
                    statementData.keys.map((key, i) => [key, statementData.scores[i]])
                ),
                
                metadata: {
                    original_debate_type: '{{ debate_type }}',
                    corrected_by: 'manual_annotation',
                    token_count: tokenData.scores.length,
                    statement_count: statementData.scores.length,
                    export_timestamp: new Date().toISOString()
                }
            };
            
            const blob = new Blob([JSON.stringify(exportData, null, 2)], 
                                 {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `manual_labels_${timestamp}.json`;
            a.click();
            URL.revokeObjectURL(url);
        }
    </script>
</body>
</html>
"""
    
    # Process transcript data
    debate_id = transcript['debate_id']
    debate_type = transcript.get('debate_type', 'unknown')
    
    # Extract context
    context_html = _extract_context_html(transcript)
    
    # Process turns with statement segmentation
    turns_data = []
    for turn in transcript.get('turns', []):
        if turn['speaker'] in ['honest', 'dishonest']:
            statements = _segment_into_statements(turn)
            
            turns_data.append({
                'speaker_label': turn.get('model_label', turn['speaker'].upper()),
                'role': turn['speaker'],
                'round': turn['turn_number'],
                'statements': statements
            })
    
    # Render template
    from jinja2 import Template
    template = Template(html_template)
    html_content = template.render(
        debate_id=debate_id,
        debate_type=debate_type,
        context_html=context_html,
        turns=turns_data,
        score_to_color=score_to_color  # Pass the function
    )
    
    # Save HTML
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Created interactive visualization: {output_path}")


def _extract_context_html(transcript: Dict) -> str:
    """Extract and format context for display"""
    debate_type = transcript.get('debate_type', 'unknown')
    problem_data = transcript.get('problem_data', {})
    model_mapping = transcript.get('model_mapping', {})
    debate_metadata = transcript.get('debate_metadata', {})
    
    # Determine which model is which
    model_a_role = model_mapping.get('MODEL_A', 'unknown')
    model_b_role = model_mapping.get('MODEL_B', 'unknown')
    
    if debate_type == 'quality':
        # For Quality debates, show story, question, and what each model argues
        question = problem_data.get('question', 'No question')
        
        # Determine who argues what based on model_a_is_honest
        model_a_is_honest = debate_metadata.get('model_a_is_honest', True)
        
        if model_a_is_honest:
            model_a_answer = problem_data.get('correct_answer', 'Unknown')
            model_b_answer = problem_data.get('incorrect_answer', 'Unknown')
        else:
            model_a_answer = problem_data.get('incorrect_answer', 'Unknown')
            model_b_answer = problem_data.get('correct_answer', 'Unknown')
        
        return f"""
        <h4>{problem_data.get('story_title', 'Story')}</h4>
        <div style="max-height: 200px; overflow-y: auto; padding: 10px; background: white; border-radius: 4px; margin-bottom: 10px;">
            {problem_data.get('story', 'No story available')}
        </div>
        <div style="background: #e3f2fd; padding: 12px; border-radius: 6px; border-left: 4px solid #2196f3;">
            <strong>Question:</strong> {question}
            
            <div style="margin-top: 12px;">
                <div style="margin-bottom: 6px;">
                    <strong>MODEL_A</strong> <span style="color: #666;">({model_a_role})</span> argues: 
                    <span style="font-weight: 600; color: #1976d2;">"{model_a_answer}"</span>
                </div>
                <div>
                    <strong>MODEL_B</strong> <span style="color: #666;">({model_b_role})</span> argues: 
                    <span style="font-weight: 600; color: #1976d2;">"{model_b_answer}"</span>
                </div>
            </div>
        </div>
        """
    
    elif debate_type == 'simple':
        topic = problem_data.get('topic', 'Unknown')
        question = problem_data.get('question', 'Unknown')
        ground_truth = problem_data.get('ground_truth', 'Unknown')
        
        # Try to extract positions from the scenario
        # The positions might be in the system prompts or explicitly stored
        model_a_position = problem_data.get('model_a_position', 'See debate transcript')
        model_b_position = problem_data.get('model_b_position', 'See debate transcript')
        
        return f"""
        <div style="background: #e3f2fd; padding: 12px; border-radius: 6px; border-left: 4px solid #2196f3;">
            <div style="margin-bottom: 8px;">
                <strong>Topic:</strong> {topic}
            </div>
            <div style="margin-bottom: 12px;">
                <strong>Question:</strong> {question}
            </div>
            
            <div style="background: white; padding: 10px; border-radius: 4px; margin-top: 10px;">
                <div style="margin-bottom: 6px;">
                    <strong>MODEL_A</strong> <span style="color: #666;">({model_a_role})</span>: 
                    <span style="font-style: italic;">{model_a_position}</span>
                </div>
                <div>
                    <strong>MODEL_B</strong> <span style="color: #666;">({model_b_role})</span>: 
                    <span style="font-style: italic;">{model_b_position}</span>
                </div>
            </div>
            
            <div style="margin-top: 10px; font-size: 13px; color: #666;">
                <strong>Ground Truth:</strong> {ground_truth}
            </div>
        </div>
        """
    
    else:  # APPS
        problem_id = problem_data.get('problem_id', 'Unknown')
        question = problem_data.get('question', 'No question')
        
        # Get solutions (truncate for display)
        honest_solution = problem_data.get('honest_solution', 'Not available')
        backdoored_solution = problem_data.get('backdoored_solution', 'Not available')
        
        # Truncate solutions for display (show first 200 chars)
        def truncate_code(code, max_len=200):
            if len(code) > max_len:
                return code[:max_len] + '...\n[truncated]'
            return code
        
        honest_preview = truncate_code(honest_solution)
        backdoored_preview = truncate_code(backdoored_solution)
        
        # Determine which model defends which solution
        model_a_is_honest = debate_metadata.get('model_a_is_honest', True)
        
        if model_a_is_honest:
            model_a_solution = honest_preview
            model_b_solution = backdoored_preview
            model_a_label = "Safe Solution"
            model_b_label = "Backdoored Solution"
        else:
            model_a_solution = backdoored_preview
            model_b_solution = honest_preview
            model_a_label = "Backdoored Solution"
            model_b_label = "Safe Solution"
        
        return f"""
        <div style="background: #e3f2fd; padding: 12px; border-radius: 6px; border-left: 4px solid #2196f3;">
            <div style="margin-bottom: 8px;">
                <strong>Problem ID:</strong> {problem_id}
            </div>
            
            <div style="margin-bottom: 12px;">
                <strong>Problem:</strong>
                <pre style="padding: 10px; background: white; border-radius: 4px; overflow-x: auto; max-height: 150px; font-size: 12px;">
{question}
                </pre>
            </div>
            
            <div style="background: white; padding: 10px; border-radius: 4px;">
                <div style="margin-bottom: 10px;">
                    <strong>MODEL_A</strong> <span style="color: #666;">({model_a_role})</span> defends: 
                    <strong style="color: #1976d2;">{model_a_label}</strong>
                    <pre style="padding: 8px; background: #f5f5f5; border-radius: 4px; margin-top: 5px; overflow-x: auto; font-size: 11px;">
{model_a_solution}
                    </pre>
                </div>
                
                <div>
                    <strong>MODEL_B</strong> <span style="color: #666;">({model_b_role})</span> defends: 
                    <strong style="color: #1976d2;">{model_b_label}</strong>
                    <pre style="padding: 8px; background: #f5f5f5; border-radius: 4px; margin-top: 5px; overflow-x: auto; font-size: 11px;">
{model_b_solution}
                    </pre>
                </div>
            </div>
        </div>
        """


def _segment_into_statements(turn: Dict) -> List[Dict]:
    """Segment a turn into statements with scores"""
    import re
    
    content = turn['content']
    tokens = turn.get('tokens', [])
    probe_scores = turn.get('probe_scores', {})
    
    if not tokens or not probe_scores:
        # No probe data - return whole content as one statement
        return [{
            'index': 0,
            'text': content,
            'token_scores': [],
            'mean_score': 0.0
        }]
    
    # Clean tokens
    clean_token_list = clean_tokens(tokens)
    
    # Get averaged probe scores
    avg_scores = []
    for i in range(len(tokens)):
        token_scores = [scores[i] for scores in probe_scores.values() if i < len(scores)]
        avg_scores.append(np.mean(token_scores) if token_scores else 0.0)
    
    # Simple sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', content.strip())
    
    statements = []
    token_idx = 0
    
    for sent_idx, sentence in enumerate(sentences):
        if not sentence:
            continue
        
        # Collect tokens for this sentence (simplified matching)
        sent_tokens = []
        sent_scores = []
        sent_length = len(sentence)
        
        while token_idx < len(clean_token_list) and len(''.join(sent_tokens)) < sent_length:
            sent_tokens.append(clean_token_list[token_idx])
            sent_scores.append(avg_scores[token_idx])
            token_idx += 1
        
        # Create statement
        statements.append({
            'index': sent_idx,
            'text': sentence,
            'token_scores': list(zip(sent_tokens, sent_scores)),
            'mean_score': np.mean(sent_scores) if sent_scores else 0.0
        })
    
    return statements