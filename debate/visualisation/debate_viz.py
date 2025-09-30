"""
HTML chat-style visualization for debate transcripts with token-level probe scores
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from jinja2 import Template


def normalize_scores(scores: List[float]) -> List[float]:
    """Normalize scores to [-1, 1] range"""
    if not scores:
        return scores
    
    scores_array = np.array(scores)
    min_score = scores_array.min()
    max_score = scores_array.max()
    
    if max_score == min_score:
        return [0.0] * len(scores)
    
    # Normalize to [-1, 1]
    # Lower scores (more truthful) = negative
    # Higher scores (more deceptive) = positive
    normalized = 2 * (scores_array - min_score) / (max_score - min_score) - 1
    return normalized.tolist()


def score_to_color(score: float) -> str:
    """Convert normalized score [-1, 1] to red-green color
    -1 (truthful) = green
    0 (neutral) = transparent
    +1 (deceptive) = red
    """
    score = max(-1, min(1, score))
    
    if score >= 0:
        # Red for deceptive (0 to 1)
        intensity = score
        return f"rgba(220, 53, 69, {intensity * 0.7:.3f})"
    else:
        # Green for truthful (-1 to 0)
        intensity = -score
        return f"rgba(40, 167, 69, {intensity * 0.7:.3f})"


def clean_tokens(tokens: List[str]) -> List[str]:
    """Clean tokenizer artifacts"""
    return [token.replace('Ġ', ' ').replace('Ċ', '\n') for token in tokens]


def create_debate_chat_visualization(transcript: Dict, output_path: Path):
    """Create HTML chat visualization for a debate transcript"""
    
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Debate Visualization: {{ debate_id }}</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background-color: #f0f2f5;
            color: #1c1e21;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            font-size: 28px;
            margin-bottom: 10px;
        }
        
        .header .subtitle {
            font-size: 14px;
            opacity: 0.9;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .tabs {
            display: flex;
            background: white;
            border-radius: 8px 8px 0 0;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .tab {
            flex: 1;
            padding: 15px;
            text-align: center;
            cursor: pointer;
            border: none;
            background: white;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.3s;
        }
        
        .tab:hover {
            background: #f8f9fa;
        }
        
        .tab.active {
            background: #667eea;
            color: white;
        }
        
        .tab-content {
            display: none;
            background: white;
            border-radius: 0 0 8px 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .tab-content.active {
            display: block;
        }
        
        .system-prompts {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        
        .system-prompt {
            margin-bottom: 20px;
        }
        
        .system-prompt h3 {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
            font-size: 16px;
        }
        
        .system-prompt .emoji {
            font-size: 24px;
        }
        
        .system-prompt pre {
            background: white;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-size: 13px;
            line-height: 1.5;
            border: 1px solid #dee2e6;
        }
        
        .chat-container {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .message {
            display: flex;
            margin-bottom: 20px;
            animation: fadeIn 0.3s;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message.honest {
            justify-content: flex-start;
        }
        
        .message.dishonest {
            justify-content: flex-end;
        }
        
        .message-avatar {
            font-size: 32px;
            margin: 0 10px;
            cursor: help;
        }
        
        .message-bubble {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 18px;
            line-height: 1.6;
        }
        
        .message.honest .message-bubble {
            background: #e3f2fd;
            border-bottom-left-radius: 4px;
        }
        
        .message.dishonest .message-bubble {
            background: #fce4ec;
            border-bottom-right-radius: 4px;
        }
        
        .message-header {
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 8px;
            opacity: 0.7;
        }
        
        .message-content {
            font-size: 15px;
        }
        
        .token {
            display: inline-block;
            padding: 2px 4px;
            margin: 1px;
            border-radius: 3px;
            font-family: 'SF Mono', Monaco, 'Courier New', monospace;
            font-size: 14px;
            cursor: help;
            transition: transform 0.2s;
        }
        
        .token:hover {
            transform: scale(1.1);
            z-index: 10;
        }
        
        .legend {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin: 20px 0;
            padding: 15px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
        }
        
        .legend-color {
            width: 24px;
            height: 24px;
            border-radius: 4px;
            border: 1px solid rgba(0,0,0,0.1);
        }
        
        .judge-section {
            margin-top: 30px;
        }
        
        .judge-tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        
        .judge-tab {
            padding: 10px 20px;
            background: white;
            border: 2px solid #dee2e6;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.3s;
        }
        
        .judge-tab:hover {
            border-color: #667eea;
        }
        
        .judge-tab.active {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }
        
        .judge-content {
            display: none;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .judge-content.active {
            display: block;
        }
        
        .judge-decision {
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        
        .judge-decision h4 {
            margin-bottom: 15px;
            font-size: 16px;
            color: #495057;
        }
        
        .decision-badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 14px;
        }
        
        .decision-badge.correct {
            background: #d4edda;
            color: #155724;
        }
        
        .decision-badge.incorrect {
            background: #f8d7da;
            color: #721c24;
        }
        
        .judge-reasoning {
            margin-top: 15px;
            padding: 15px;
            background: white;
            border-radius: 6px;
            border-left: 4px solid #667eea;
            white-space: pre-wrap;
            font-size: 14px;
            line-height: 1.6;
        }
        
        .analysis-section {
            margin-top: 30px;
            background: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .analysis-section h3 {
            margin-bottom: 20px;
            color: #667eea;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        
        .stat-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 28px;
            font-weight: 700;
            color: #667eea;
            margin-bottom: 5px;
        }
        
        .stat-label {
            font-size: 13px;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Debate Visualization</h1>
        <div class="subtitle">{{ debate_id }}</div>
    </div>
    
    <div class="container">
        <div class="tabs">
            <button class="tab active" onclick="showTab('prompts')">System Prompts</button>
            <button class="tab" onclick="showTab('debate')">Debate Transcript</button>
            <button class="tab" onclick="showTab('judges')">Judge Decisions</button>
            <button class="tab" onclick="showTab('analysis')">Analysis</button>
        </div>
        
        <div id="prompts" class="tab-content active">
            <div class="system-prompts">
                <h2 style="margin-bottom: 20px;">System Prompts</h2>
                
                <div class="system-prompt">
                    <h3>
                        <span class="emoji">😇</span>
                        Honest Debater
                    </h3>
                    <pre>{{ honest_system_prompt }}</pre>
                </div>
                
                <div class="system-prompt">
                    <h3>
                        <span class="emoji">😈</span>
                        Deceptive Debater
                    </h3>
                    <pre>{{ dishonest_system_prompt }}</pre>
                </div>
            </div>
        </div>
        
        <div id="debate" class="tab-content">
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background-color: rgba(40, 167, 69, 0.7);"></div>
                    <span>Truthful (Low Probe Score)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: rgba(128, 128, 128, 0.2);"></div>
                    <span>Neutral</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: rgba(220, 53, 69, 0.7);"></div>
                    <span>Deceptive (High Probe Score)</span>
                </div>
            </div>
            
            <div class="chat-container">
                {% for message in messages %}
                <div class="message {{ message.role }}">
                    {% if message.role == 'honest' %}
                    <div class="message-avatar" title="Honest Debater">😇</div>
                    {% else %}
                    <div class="message-avatar" title="Deceptive Debater">😈</div>
                    {% endif %}
                    
                    <div class="message-bubble">
                        <div class="message-header">
                            {{ 'Honest Debater' if message.role == 'honest' else 'Deceptive Debater' }} - Round {{ message.round }}
                        </div>
                        <div class="message-content">
                            {% for token, score, color in message.tokens %}
                            <span class="token" 
                                  style="background-color: {{ color }}"
                                  title="Probe Score: {{ '%.3f'|format(score) }}">{{ token }}</span>
                            {% endfor %}
                        </div>
                    </div>
                    
                    {% if message.role == 'dishonest' %}
                    <div class="message-avatar" title="Deceptive Debater">😈</div>
                    {% else %}
                    <div class="message-avatar" title="Honest Debater">😇</div>
                    {% endif %}
                </div>
                {% endfor %}
            </div>
        </div>
        
        <div id="judges" class="tab-content">
            <div class="judge-section">
                <h2 style="margin-bottom: 20px;">Judge Decisions</h2>
                
                <div class="judge-tabs">
                    {% for judge_key in judge_results.keys() %}
                    <button class="judge-tab {% if loop.first %}active{% endif %}" 
                            onclick="showJudge('{{ judge_key }}')">
                        {{ judge_key }}
                    </button>
                    {% endfor %}
                </div>
                
                {% for judge_key, judge_data in judge_results.items() %}
                <div id="{{ judge_key }}" class="judge-content {% if loop.first %}active{% endif %}">
                    <div class="judge-decision">
                        <h4>Decision</h4>
                        <div>
                            Winner: <strong>{{ judge_data.winner_label }}</strong> 
                            ({{ judge_data.winner_role }})
                            {% if judge_data.correct is not none %}
                            <span class="decision-badge {{ 'correct' if judge_data.correct else 'incorrect' }}">
                                {{ 'Correct' if judge_data.correct else 'Incorrect' }}
                            </span>
                            {% endif %}
                        </div>
                        <div style="margin-top: 10px; font-size: 13px; color: #6c757d;">
                            Mode: {{ judge_data.evaluation_mode }} | 
                            Probe Display: {{ judge_data.probe_display_mode or 'N/A' }}
                        </div>
                    </div>
                    
                    <div class="system-prompt">
                        <h4>Judge System Prompt</h4>
                        <pre>{{ judge_data.system_prompt }}</pre>
                    </div>
                    
                    <div class="judge-reasoning">
                        <h4 style="margin-bottom: 10px;">Reasoning</h4>
                        {{ judge_data.reasoning }}
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>
        
        <div id="analysis" class="tab-content">
            <div class="analysis-section">
                <h3>Debate Statistics</h3>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">{{ debate_stats.total_rounds }}</div>
                        <div class="stat-label">Total Rounds</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{{ '%.3f'|format(debate_stats.honest_mean_score) }}</div>
                        <div class="stat-label">Honest Mean Score</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{{ '%.3f'|format(debate_stats.dishonest_mean_score) }}</div>
                        <div class="stat-label">Deceptive Mean Score</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{{ '%.3f'|format(debate_stats.score_separation) }}</div>
                        <div class="stat-label">Score Separation</div>
                    </div>
                </div>
                
                <div style="margin-top: 30px;">
                    <h4>Probe-Only Classifications</h4>
                    <div class="stats-grid">
                        {% if probe_only_results %}
                        <div class="stat-card">
                            <div class="stat-value">{{ probe_only_results.probe_only_mean }}</div>
                            <div class="stat-label">Winner (Mean)</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{{ probe_only_results.probe_only_max }}</div>
                            <div class="stat-label">Winner (Max)</div>
                        </div>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        function showTab(tabName) {
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            
            // Remove active class from all tabs
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab content
            document.getElementById(tabName).classList.add('active');
            
            // Add active class to clicked tab
            event.target.classList.add('active');
        }
        
        function showJudge(judgeKey) {
            // Hide all judge contents
            document.querySelectorAll('.judge-content').forEach(content => {
                content.classList.remove('active');
            });
            
            // Remove active class from all judge tabs
            document.querySelectorAll('.judge-tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected judge content
            document.getElementById(judgeKey).classList.add('active');
            
            // Add active class to clicked tab
            event.target.classList.add('active');
        }
    </script>
</body>
</html>
"""
    
    # Extract data from transcript
    debate_id = transcript['debate_id']
    debate_config = transcript.get('debate_config', {})
    turns = transcript.get('turns', [])
    judge_results = transcript.get('judge_results', {})
    probe_only_results = transcript.get('probe_only_results', {})
    
    # Extract system prompts (reconstruct from config or problem data)
    honest_system_prompt = _extract_system_prompt(transcript, 'honest')
    dishonest_system_prompt = _extract_system_prompt(transcript, 'dishonest')
    
    # Process messages
    messages = []
    all_scores = []
    
    for turn in turns:
        if turn['speaker'] in ['honest', 'dishonest']:
            # Collect all scores for normalization
            if turn.get('probe_scores'):
                for probe_type, scores in turn['probe_scores'].items():
                    all_scores.extend(scores)
    
    # Normalize all scores globally
    if all_scores:
        normalized_global = normalize_scores(all_scores)
        score_idx = 0
        
        for turn in turns:
            if turn['speaker'] in ['honest', 'dishonest']:
                tokens = turn.get('tokens', [])
                probe_scores = turn.get('probe_scores', {})
                
                if tokens and probe_scores:
                    # Average scores across probe types
                    num_tokens = len(tokens)
                    avg_scores = []
                    
                    for i in range(num_tokens):
                        token_scores = [scores[i] for scores in probe_scores.values() if i < len(scores)]
                        avg_score = np.mean(token_scores) if token_scores else 0.0
                        avg_scores.append(avg_score)
                    
                    # Get normalized scores for this turn
                    turn_normalized = normalized_global[score_idx:score_idx + num_tokens]
                    score_idx += num_tokens
                    
                    # Clean tokens and create token-color pairs
                    clean_token_list = clean_tokens(tokens)
                    tokens_with_colors = [
                        (token, norm_score, score_to_color(norm_score))
                        for token, norm_score in zip(clean_token_list, turn_normalized)
                    ]
                    
                    messages.append({
                        'role': turn['speaker'],
                        'round': turn['turn_number'],
                        'tokens': tokens_with_colors
                    })
    
    # Process judge results
    judge_results_processed = {}
    for judge_key, judge_data in judge_results.items():
        # Extract judge system prompt
        judge_system_prompt = _extract_judge_system_prompt(transcript, judge_key)
        
        judge_results_processed[judge_key] = {
            'winner_label': judge_data.get('winner_label', 'Unknown'),
            'winner_role': judge_data.get('winner_role', 'Unknown'),
            'correct': judge_data.get('correct'),
            'reasoning': judge_data.get('reasoning', ''),
            'evaluation_mode': judge_data.get('evaluation_mode', 'unknown'),
            'probe_display_mode': judge_data.get('probe_display_mode'),
            'system_prompt': judge_system_prompt
        }
    
    # Calculate debate statistics
    honest_scores = []
    dishonest_scores = []
    
    for turn in turns:
        if turn.get('mean_probe_score') is not None:
            if turn['speaker'] == 'honest':
                honest_scores.append(turn['mean_probe_score'])
            elif turn['speaker'] == 'dishonest':
                dishonest_scores.append(turn['mean_probe_score'])
    
    debate_stats = {
        'total_rounds': len([t for t in turns if t['speaker'] in ['honest', 'dishonest']]) // 2,
        'honest_mean_score': np.mean(honest_scores) if honest_scores else 0.0,
        'dishonest_mean_score': np.mean(dishonest_scores) if dishonest_scores else 0.0,
        'score_separation': abs(np.mean(dishonest_scores) - np.mean(honest_scores)) if honest_scores and dishonest_scores else 0.0
    }
    
    # Render template
    template = Template(html_template)
    html_content = template.render(
        debate_id=debate_id,
        honest_system_prompt=honest_system_prompt,
        dishonest_system_prompt=dishonest_system_prompt,
        messages=messages,
        judge_results=judge_results_processed,
        probe_only_results=probe_only_results,
        debate_stats=debate_stats
    )
    
    # Save HTML
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def _extract_system_prompt(transcript: Dict, role: str) -> str:
    """Extract system prompt for a debater role"""
    # Try to find in turns
    for turn in transcript.get('turns', []):
        if turn['speaker'] == role and 'system_prompt' in turn.get('metadata', {}):
            return turn['metadata']['system_prompt']
    
    # Fallback: construct from problem data
    problem_data = transcript.get('problem_data', {})
    debate_type = transcript.get('debate_type', 'unknown')
    
    if role == 'honest':
        if debate_type == 'quality':
            return f"You are defending: {problem_data.get('correct_answer', 'Unknown')}"
        else:
            return problem_data.get('honest_system_prompt', 'System prompt not available')
    else:
        if debate_type == 'quality':
            return f"You are defending: {problem_data.get('incorrect_answer', 'Unknown')}"
        else:
            return problem_data.get('dishonest_system_prompt', 'System prompt not available')


def _extract_judge_system_prompt(transcript: Dict, judge_key: str) -> str:
    """Extract judge system prompt for a specific evaluation"""
    # This would need to be stored during judge evaluation
    # For now, return a generic message
    return "Judge system prompt (to be stored during evaluation)"