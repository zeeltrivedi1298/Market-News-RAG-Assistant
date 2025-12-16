import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field

@dataclass
class ProcessingStep:
    step_number: int
    name: str
    description: str
    status: str  # "started", "completed", "error"
    details: Dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    duration: Optional[float] = None
    
    def to_html(self) -> str:
        status_config = {
            "started": {"icon": "⏳", "color": "#FFA500", "bg": "#FFF4E6", "border": "#FFA500"},
            "completed": {"icon": "✅", "color": "#10B981", "bg": "#ECFDF5", "border": "#10B981"},
            "error": {"icon": "❌", "color": "#EF4444", "bg": "#FEF2F2", "border": "#EF4444"}
        }
        
        config = status_config.get(self.status, status_config["started"])
        
        html = f"""
        <div style="
            margin-bottom: 12px;
            padding: 14px;
            border-left: 4px solid {config['border']};
            background-color: {config['bg']};
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        ">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 20px;">{config['icon']}</span>
                    <div>
                        <strong style="color: {config['color']}; font-size: 15px;">Step {self.step_number}: {self.name}</strong>
                        <div style="color: #6B7280; font-size: 13px; margin-top: 2px;">{self.description}</div>
                    </div>
                </div>
                {f'<span style="color: {config["color"]}; font-weight: 600; font-size: 12px;">{self.duration:.2f}s</span>' if self.duration else ''}
            </div>
        """
        
        if self.details:
            html += '<div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(0,0,0,0.1);">'
            html += '<div style="font-size: 12px; color: #4B5563;"><strong>Details:</strong></div>'
            html += '<div style="margin-top: 6px; display: flex; flex-wrap: wrap; gap: 8px;">'
            
            for key, value in self.details.items():
                if key == "error":
                    html += f'''
                    <div style="
                        padding: 6px 10px;
                        background-color: #FEE2E2;
                        border-radius: 4px;
                        color: #991B1B;
                        font-size: 12px;
                        font-weight: 500;
                    ">
                        <strong>Error:</strong> {value}
                    </div>
                    '''
                elif isinstance(value, list) and len(value) > 0:
                    display_value = ', '.join(str(v) for v in value[:5]) if len(value) <= 5 else f"{len(value)} items"
                    html += f'''
                    <div style="
                        padding: 4px 8px;
                        background-color: rgba(0,0,0,0.05);
                        border-radius: 4px;
                        font-size: 11px;
                        color: #374151;
                    ">
                        <strong>{key.replace('_', ' ').title()}:</strong> {display_value}
                    </div>
                    '''
                elif isinstance(value, bool):
                    html += f'''
                    <div style="
                        padding: 4px 8px;
                        background-color: {'#D1FAE5' if value else '#FEE2E2'};
                        border-radius: 4px;
                        font-size: 11px;
                        color: #065F46;
                    ">
                        <strong>{key.replace('_', ' ').title()}:</strong> {'Yes' if value else 'No'}
                    </div>
                    '''
                else:
                    html += f'''
                    <div style="
                        padding: 4px 8px;
                        background-color: rgba(0,0,0,0.05);
                        border-radius: 4px;
                        font-size: 11px;
                        color: #374151;
                    ">
                        <strong>{key.replace('_', ' ').title()}:</strong> {str(value)[:100] + '...' if len(str(value)) > 100 else str(value)}
                    </div>
                    '''
            
            html += '</div></div>'
        
        html += '</div>'
        return html

class ProcessingStepsTracker:
    def __init__(self):
        self.steps: List[ProcessingStep] = []
        self.step_counter = 0
        self.start_times: Dict[int, float] = {}
    
    def start_step(self, name: str, description: str, details: Optional[Dict] = None) -> int:
        self.step_counter += 1
        step = ProcessingStep(
            step_number=self.step_counter,
            name=name,
            description=description,
            status="started",
            details=details or {}
        )
        self.steps.append(step)
        self.start_times[self.step_counter] = time.time()
        return self.step_counter
    
    def complete_step(self, step_number: int, details: Optional[Dict] = None):
        if step_number <= len(self.steps):
            step = self.steps[step_number - 1]
            step.status = "completed"
            if details:
                step.details.update(details)
            if step_number in self.start_times:
                step.duration = time.time() - self.start_times[step_number]
                del self.start_times[step_number]
    
    def error_step(self, step_number: int, error_message: str, details: Optional[Dict] = None):
        if step_number <= len(self.steps):
            step = self.steps[step_number - 1]
            step.status = "error"
            step.details["error"] = error_message
            if details:
                step.details.update(details)
            if step_number in self.start_times:
                step.duration = time.time() - self.start_times[step_number]
                del self.start_times[step_number]
    
    def to_html(self) -> str:
        if not self.steps:
            return """
            <div style="
                padding: 20px;
                text-align: center;
                color: #6B7280;
                background-color: #F9FAFB;
                border-radius: 8px;
            ">
                <div style="font-size: 48px; margin-bottom: 10px;">⏳</div>
                <div style="font-size: 14px;">No steps recorded yet.</div>
            </div>
            """
        
        completed_steps = sum(1 for s in self.steps if s.status == "completed")
        total_duration = sum(step.duration for step in self.steps if step.duration)
        
        html = f"""
        <div style="
            margin-bottom: 20px;
            padding: 16px;
            background: linear-gradient(135deg, #C4B5FD 0%, #A78BFA 100%);
            border-radius: 8px;
            color: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 18px; font-weight: 600; margin-bottom: 4px;">🔄 Processing Steps</div>
                    <div style="font-size: 13px; opacity: 0.9;">{completed_steps} of {len(self.steps)} steps completed</div>
                </div>
                {f'<div style="text-align: right;"><div style="font-size: 24px; font-weight: 700;">{total_duration:.2f}s</div><div style="font-size: 11px; opacity: 0.9;">Total Duration</div></div>' if total_duration else ''}
            </div>
        </div>
        """
        
        html += '<div style="display: flex; flex-direction: column; gap: 8px;">'
        for step in self.steps:
            html += step.to_html()
        html += '</div>'
        
        return html
    
    def to_markdown(self) -> str:
        return self.to_html()
