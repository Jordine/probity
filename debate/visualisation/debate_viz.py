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


def score_to_color(score: float, threshold: float = 0.5) -> str:
    """Convert score [0, 1] to color based on threshold
    Above threshold = red (deceptive)
    Below threshold = green (honest)
    Opacity based on distance from threshold
    """
    distance = abs(score - threshold)
    max_opacity = 0.7
    min_opacity = 0.1
    
    # Scale opacity based on distance from threshold (closer = more transparent)
    opacity = min_opacity + (max_opacity - min_opacity) * min(distance * 2, 1)
    
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
                    <div class="turn" data-turn-id="{{ loop.index0 }}">
                        <div class="speaker-label">
                            {{ turn.speaker_label }}
                            <span class="role-badge {{ turn.role }}">{{ turn.role.upper() }}</span>
                            - Round {{ turn.round }}
                        </div>
                        
                        {% for statement in turn.statements %}
                        <div class="statement" 
                             data-turn="{{ loop.index0 }}"
                             data-statement="{{ statement.index }}"
                             data-role="{{ turn.role }}"
                             data-score="{{ statement.mean_score }}"
                             onclick="toggleLabel(this)">
                            <div class="statement-tokens">
                                {% for token, score in statement.token_scores %}
                                <span class="token" 
                                      data-score="{{ score }}"
                                      style="background-color: {{ score_to_color(score, 0.5) }}">{{ token }}</span>
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
                    <label>Threshold: <span id="threshold-value">0.500</span></label>
                    <input type="range" class="slider" id="threshold-slider" 
                           min="0" max="1" step="0.001" value="0.5">
                </div>


                <button class="export-btn" id="flip-btn" onclick="flipProbeClasses()" 
                        style="margin-top: 10px; background: #e74c3c;">
                    🔄 Flip Probe Classes
                </button>
                
                
                <div class="metrics">
                    <h4>📊 Live Metrics</h4>
                    <div id="metrics-display">
                        <div>Probe AUROC: <span id="auroc-value">-</span></div>
                        <div>Accuracy @ Threshold: <span id="accuracy-value">-</span></div>
                        <div>Statements Above: <span id="above-count">-</span></div>
                    </div>
                </div>
                
                <button class="export-btn" onclick="exportLabels()">
                    💾 Export Corrected Labels
                </button>
                
                <div style="margin-top: 15px; font-size: 12px; color: #666;">
                    <p>🖱️ Click statements to toggle label</p>
                    <p>🟢 Green = Honest</p>
                    <p>🔴 Red = Deceptive</p>
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
                <h4>ROC Curve</h4>
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
        // Global state
        let statementLabels = {};  // Maps statement ID to label (0=honest, 1=deceptive)
        let currentThreshold = 0.5;
        let tokenDistChart, statementDistChart, rocChart;
        let isFlipped = false; 


        // Helper function to get effective score based on flip state
        function getEffectiveScore(score) {
            return isFlipped ? (1 - score) : score;
        }
        
        // Flip probe class interpretation
        function flipProbeClasses() {
            isFlipped = !isFlipped;
            const btn = document.getElementById('flip-btn');
            btn.textContent = isFlipped ? '🔄 Un-flip Probe Classes' : '🔄 Flip Probe Classes';
            btn.style.background = isFlipped ? '#2ecc71' : '#e74c3c';
            
            // Update everything
            updateColors();
            updateMetrics();
            updatePlots();
        }
        
        
        // Initialize labels from original roles
        document.querySelectorAll('.statement').forEach(stmt => {
            const key = `${stmt.dataset.turn}-${stmt.dataset.statement}`;
            // Initial labeling: dishonest = deceptive (1), honest = honest (0)
            statementLabels[key] = stmt.dataset.role === 'dishonest' ? 1 : 0;
            updateStatementStyle(stmt);
        });
        
        // Threshold slider
        document.getElementById('threshold-slider').addEventListener('input', (e) => {
            currentThreshold = parseFloat(e.target.value);
            document.getElementById('threshold-value').textContent = currentThreshold.toFixed(3);
            updateColors();
            updateMetrics();
            updatePlots();
        });
        
        function toggleLabel(element) {
            const key = `${element.dataset.turn}-${element.dataset.statement}`;
            statementLabels[key] = 1 - statementLabels[key];  // Toggle between 0 and 1
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
                const score = getEffectiveScore(rawScore);  // MODIFIED: Use effective score
                const distance = Math.abs(score - currentThreshold);
                const maxOpacity = 0.7;
                const minOpacity = 0.1;
                const opacity = minOpacity + (maxOpacity - minOpacity) * Math.min(distance * 2, 1);
                
                if (score >= currentThreshold) {
                    token.style.backgroundColor = `rgba(220, 53, 69, ${opacity})`;
                } else {
                    token.style.backgroundColor = `rgba(40, 167, 69, ${opacity})`;
                }
            });
        }
        
        function calculateAUROC(labels, scores) {
            // Simple AUROC calculation
            const pairs = labels.map((label, i) => ({label, score: scores[i]}));
            pairs.sort((a, b) => a.score - b.score);
            
            let auc = 0;
            let tpr_prev = 0;
            let fpr_prev = 0;
            
            const n_pos = labels.filter(l => l === 1).length;
            const n_neg = labels.filter(l => l === 0).length;
            
            if (n_pos === 0 || n_neg === 0) return 0.5;
            
            for (let i = 0; i < pairs.length; i++) {
                const tp = pairs.slice(i).filter(p => p.label === 1).length;
                const fp = pairs.slice(i).filter(p => p.label === 0).length;
                
                const tpr = tp / n_pos;
                const fpr = fp / n_neg;
                
                auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2;
                
                tpr_prev = tpr;
                fpr_prev = fpr;
            }
            
            return auc;
        }
        
        function getROCPoints(labels, scores) {
            const thresholds = [...new Set(scores)].sort((a, b) => b - a);
            const points = [];
            
            const n_pos = labels.filter(l => l === 1).length;
            const n_neg = labels.filter(l => l === 0).length;
            
            if (n_pos === 0 || n_neg === 0) {
                return [{x: 0, y: 0}, {x: 1, y: 1}];
            }
            
            thresholds.forEach(thresh => {
                const predictions = scores.map(s => s >= thresh ? 1 : 0);
                const tp = predictions.filter((p, i) => p === 1 && labels[i] === 1).length;
                const fp = predictions.filter((p, i) => p === 1 && labels[i] === 0).length;
                
                points.push({
                    x: fp / n_neg,  // FPR
                    y: tp / n_pos,  // TPR
                    threshold: thresh
                });
            });
            
            // Add endpoints
            points.unshift({x: 1, y: 1, threshold: 0});
            points.push({x: 0, y: 0, threshold: 1});
            
            return points;
        }
        
        
        function updateMetrics() {
            const labels = [];
            const scores = [];
            
            document.querySelectorAll('.statement').forEach(stmt => {
                const key = `${stmt.dataset.turn}-${stmt.dataset.statement}`;
                labels.push(statementLabels[key]);
                const rawScore = parseFloat(stmt.dataset.score);
                scores.push(getEffectiveScore(rawScore));  // MODIFIED: Use effective score
            });
            
            const auroc = calculateAUROC(labels, scores);
            
            // Calculate accuracy at current threshold
            const predictions = scores.map(s => s >= currentThreshold ? 1 : 0);
            const correct = predictions.filter((p, i) => p === labels[i]).length;
            const accuracy = correct / predictions.length;
            
            const aboveCount = scores.filter(s => s >= currentThreshold).length;
            
            document.getElementById('auroc-value').textContent = auroc.toFixed(3);
            document.getElementById('accuracy-value').textContent = accuracy.toFixed(3);
            document.getElementById('above-count').textContent = `${aboveCount}/${scores.length}`;
        }

        function updatePlots() {
            // Collect all data
            const statementLabels = [];
            const statementScores = [];
            const tokenLabels = [];
            const tokenScores = [];
            
            document.querySelectorAll('.statement').forEach(stmt => {
                const key = `${stmt.dataset.turn}-${stmt.dataset.statement}`;
                const label = statementLabels[key];
                const rawScore = parseFloat(stmt.dataset.score);
                const score = getEffectiveScore(rawScore);  // MODIFIED: Use effective score
                
                statementLabels.push(label);
                statementScores.push(score);
                
                // Collect token scores
                stmt.querySelectorAll('.token').forEach(token => {
                    tokenLabels.push(label);
                    const rawTokenScore = parseFloat(token.dataset.score);
                    tokenScores.push(getEffectiveScore(rawTokenScore));  // MODIFIED
                });
            });
            
            // Update token distribution
            updateDistributionChart(tokenDistChart, tokenLabels, tokenScores, 'Token');
            
            // Update statement distribution
            updateDistributionChart(statementDistChart, statementLabels, statementScores, 'Statement');
            
            // Update ROC curve
            updateROCChart(statementLabels, statementScores);
            
            // Update statistics
            updateStatistics(statementLabels, statementScores, tokenLabels, tokenScores);
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
            chart.update();
        }
        
        function updateROCChart(labels, scores) {
            const points = getROCPoints(labels, scores);
            const auroc = calculateAUROC(labels, scores);
            
            // Find current threshold point
            const currentPoint = points.find(p => 
                Math.abs(p.threshold - currentThreshold) < 0.01) || points[0];
            
            rocChart.data.datasets[0].data = points;
            rocChart.data.datasets[1].data = [currentPoint];
            rocChart.data.datasets[0].label = `ROC (AUC = ${auroc.toFixed(3)})`;
            rocChart.update();
        }
        
        function updateStatistics(stmtLabels, stmtScores, tokLabels, tokScores) {
            const stmtHonest = stmtScores.filter((s, i) => stmtLabels[i] === 0);
            const stmtDeceptive = stmtScores.filter((s, i) => stmtLabels[i] === 1);
            
            const html = `
                <div style="margin-bottom: 10px;">
                    <strong>Statement-Level</strong><br>
                    Honest: μ=${mean(stmtHonest).toFixed(3)}, σ=${std(stmtHonest).toFixed(3)}<br>
                    Deceptive: μ=${mean(stmtDeceptive).toFixed(3)}, σ=${std(stmtDeceptive).toFixed(3)}<br>
                    Separation: ${(mean(stmtDeceptive) - mean(stmtHonest)).toFixed(3)}
                </div>
                <div>
                    <strong>Token-Level</strong><br>
                    Total tokens: ${tokScores.length}<br>
                    Honest: ${tokLabels.filter(l => l === 0).length}<br>
                    Deceptive: ${tokLabels.filter(l => l === 1).length}
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
            const exportData = {
                debate_id: '{{ debate_id }}',
                timestamp: timestamp,
                threshold: currentThreshold,
                labels: statementLabels,
                statement_scores: {},
                metadata: {
                    original_debate_type: '{{ debate_type }}',
                    corrected_by: 'manual_annotation'
                }
            };
            
            // Add scores
            document.querySelectorAll('.statement').forEach(stmt => {
                const key = `${stmt.dataset.turn}-${stmt.dataset.statement}`;
                exportData.statement_scores[key] = parseFloat(stmt.dataset.score);
            });
            
            // Create download
            const blob = new Blob([JSON.stringify(exportData, null, 2)], 
                                 {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `corrected_labels_${timestamp}.json`;
            a.click();
            URL.revokeObjectURL(url);
        }
        
        // Initialize charts
        window.addEventListener('DOMContentLoaded', () => {
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
                        tension: 0
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
            
            // Initial update
            updateMetrics();
            updatePlots();
        });
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
    
    if debate_type == 'quality':
        return f"""
        <h4>{problem_data.get('story_title', 'Story')}</h4>
        <div style="max-height: 200px; overflow-y: auto; padding: 10px; background: white; border-radius: 4px;">
            {problem_data.get('story', 'No story available')}
        </div>
        <div style="margin-top: 10px;">
            <strong>Question:</strong> {problem_data.get('question', 'No question')}
        </div>
        """
    elif debate_type == 'simple':
        return f"""
        <div><strong>Topic:</strong> {problem_data.get('topic', 'Unknown')}</div>
        <div><strong>Question:</strong> {problem_data.get('question', 'Unknown')}</div>
        <div><strong>Ground Truth:</strong> {problem_data.get('ground_truth', 'Unknown')}</div>
        """
    else:  # APPS
        return f"""
        <div><strong>Problem ID:</strong> {problem_data.get('problem_id', 'Unknown')}</div>
        <div><strong>Question:</strong></div>
        <pre style="padding: 10px; background: white; border-radius: 4px; overflow-x: auto;">
{problem_data.get('question', 'No question')}
        </pre>
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
            'mean_score': 0.5
        }]
    
    # Clean tokens
    clean_token_list = clean_tokens(tokens)
    
    # Get averaged probe scores
    avg_scores = []
    for i in range(len(tokens)):
        token_scores = [scores[i] for scores in probe_scores.values() if i < len(scores)]
        avg_scores.append(np.mean(token_scores) if token_scores else 0.5)
    
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
            'mean_score': np.mean(sent_scores) if sent_scores else 0.5
        })
    
    return statements