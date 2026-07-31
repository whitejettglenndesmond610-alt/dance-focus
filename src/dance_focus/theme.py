APP_STYLESHEET = """
QMainWindow, QWidget#appRoot {
    background: #090b0e;
    color: #eef1f5;
    font-family: "Inter", "Noto Sans CJK SC", "Noto Sans", sans-serif;
    font-size: 13px;
}

QFrame#topBar {
    background: #0f1216;
    border: 1px solid #20252c;
    border-radius: 14px;
}

QLabel#brandMark {
    background: #d9ff57;
    color: #090b0e;
    border-radius: 9px;
    font-size: 15px;
    font-weight: 900;
}

QLabel#brandTitle {
    color: #f7f9fb;
    font-size: 16px;
    font-weight: 800;
    letter-spacing: 1px;
}

QLabel#eyebrow {
    color: #68717d;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 2px;
}

QLabel#projectName {
    color: #cdd2d9;
    font-family: "JetBrains Mono", "Noto Sans Mono", monospace;
    font-size: 11px;
}

QLabel#runtimeChip {
    background: #161b20;
    border: 1px solid #2a3139;
    border-radius: 10px;
    color: #aab2bd;
    padding: 7px 11px;
    font-size: 10px;
}

QFrame#stageCard, QFrame#inspectorCard, QFrame#sectionCard, QFrame#statusCard {
    background: #0f1216;
    border: 1px solid #20252c;
    border-radius: 14px;
}

QFrame#monitorHeader, QFrame#transportBar {
    background: transparent;
    border: none;
}

QLabel#panelTitle {
    color: #f4f6f8;
    font-size: 14px;
    font-weight: 750;
}

QLabel#sectionNumber {
    color: #d9ff57;
    font-family: "JetBrains Mono", "Noto Sans Mono", monospace;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
}

QLabel#sectionTitle {
    color: #eef1f5;
    font-size: 13px;
    font-weight: 700;
}

QLabel#muted, QLabel#sourceInfo, QLabel#videoMeta {
    color: #727b87;
    font-size: 11px;
}

QLabel#videoMeta, QLabel#timecode, QLabel#valuePill {
    font-family: "JetBrains Mono", "Noto Sans Mono", monospace;
}

QLabel#timecode {
    color: #f1f3f6;
    background: #090c0f;
    border: 1px solid #222832;
    border-radius: 8px;
    padding: 7px 10px;
    font-size: 11px;
}

QLabel#valuePill {
    background: #1a1f25;
    border: 1px solid #2a313a;
    border-radius: 8px;
    color: #d9ff57;
    padding: 4px 8px;
    font-size: 10px;
}

QFrame#segmentedControl {
    background: #090c10;
    border: 1px solid #242a32;
    border-radius: 10px;
}

QFrame#segmentIndicator {
    background: #252b33;
    border: 1px solid #39414c;
    border-radius: 7px;
}

QFrame#segmentedControl QToolButton {
    background: transparent;
    border: none;
    color: #727b87;
    padding: 5px 9px;
    font-size: 10px;
    font-weight: 650;
}

QFrame#segmentedControl QToolButton:checked {
    color: #f7f8fa;
}

QFrame#segmentedControl QToolButton:disabled {
    color: #3d444e;
}

QSlider::groove:horizontal {
    height: 4px;
    background: #292f38;
    border-radius: 2px;
}

QSlider::sub-page:horizontal {
    background: #d9ff57;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    width: 15px;
    height: 15px;
    margin: -6px 0;
    background: #f3f6f8;
    border: 3px solid #d9ff57;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    background: #ffffff;
    border-color: #edff9b;
}

QProgressBar {
    min-height: 7px;
    max-height: 7px;
    background: #20262e;
    border: none;
    border-radius: 3px;
    text-align: center;
    color: transparent;
}

QProgressBar::chunk {
    background: #d9ff57;
    border-radius: 3px;
}

QScrollArea, QScrollArea > QWidget > QWidget {
    background: transparent;
    border: none;
}

QScrollBar:vertical {
    background: transparent;
    width: 7px;
    margin: 4px 0;
}

QScrollBar::handle:vertical {
    background: #303741;
    min-height: 32px;
    border-radius: 3px;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QToolTip {
    background: #20252c;
    color: #f5f7fa;
    border: 1px solid #3a424d;
    padding: 6px;
}
"""
