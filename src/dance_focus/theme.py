APP_STYLESHEET = """
QMainWindow, QWidget#appRoot {
    background: #f4f8f6;
    color: #1c2925;
    font-family: "Inter", "Noto Sans CJK SC", "Noto Sans", sans-serif;
    font-size: 13px;
}

QFrame#topBar {
    background: #ffffff;
    border: 1px solid #dce8e3;
    border-radius: 12px;
}

QLabel#brandMark {
    background: #42bd91;
    color: #ffffff;
    border-radius: 9px;
    font-size: 15px;
    font-weight: 900;
}

QLabel#brandTitle {
    color: #172520;
    font-size: 17px;
    font-weight: 800;
}

QLabel#eyebrow {
    color: #7c8d87;
    font-size: 10px;
    font-weight: 650;
}

QLabel#projectName {
    color: #52645e;
    font-size: 11px;
}

QLabel#runtimeChip {
    background: #eef7f3;
    border: 1px solid #d6e9e1;
    border-radius: 10px;
    color: #527168;
    padding: 7px 11px;
    font-size: 10px;
}

QFrame#stageCard, QFrame#inspectorCard, QFrame#sectionCard {
    background: #ffffff;
    border: 1px solid #dce8e3;
    border-radius: 12px;
}

QFrame#statusCard {
    background: #eaf7f2;
    border: 1px solid #d1e9df;
    border-radius: 11px;
}

QFrame#advancedPanel {
    background: #f7faf9;
    border: 1px solid #e2ece8;
    border-radius: 10px;
}

QFrame#monitorHeader, QFrame#transportBar {
    background: transparent;
    border: none;
}

QLabel#panelTitle {
    color: #172520;
    font-size: 14px;
    font-weight: 750;
}

QLabel#sectionNumber {
    color: #299a73;
    font-size: 10px;
    font-weight: 700;
}

QLabel#sectionTitle {
    color: #1b2a25;
    font-size: 14px;
    font-weight: 750;
}

QLabel#muted, QLabel#sourceInfo, QLabel#videoMeta {
    color: #71817b;
    font-size: 11px;
}

QLabel#videoMeta, QLabel#timecode, QLabel#valuePill {
    font-family: "JetBrains Mono", "Noto Sans Mono", monospace;
}

QLabel#timecode {
    color: #42554f;
    background: #f3f7f5;
    border: 1px solid #dce7e3;
    border-radius: 8px;
    padding: 7px 10px;
    font-size: 11px;
}

QLabel#valuePill {
    background: #e7f6f0;
    border: 1px solid #d0e9df;
    border-radius: 8px;
    color: #248361;
    padding: 4px 8px;
    font-size: 10px;
}

QFrame#segmentedControl {
    background: #eef3f1;
    border: 1px solid #dbe6e2;
    border-radius: 10px;
}

QFrame#segmentIndicator {
    background: #ffffff;
    border: 1px solid #cfe3db;
    border-radius: 7px;
}

QFrame#segmentedControl QToolButton {
    background: transparent;
    border: none;
    color: #71817b;
    padding: 5px 9px;
    font-size: 10px;
    font-weight: 650;
}

QFrame#segmentedControl QToolButton:checked {
    color: #23825f;
}

QFrame#segmentedControl QToolButton:disabled {
    color: #b3beba;
}

QSlider::groove:horizontal {
    height: 4px;
    background: #dce7e3;
    border-radius: 2px;
}

QSlider::sub-page:horizontal {
    background: #42bd91;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    width: 15px;
    height: 15px;
    margin: -6px 0;
    background: #ffffff;
    border: 3px solid #42bd91;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    border-color: #269e75;
}

QProgressBar {
    min-height: 7px;
    max-height: 7px;
    background: #d5e7df;
    border: none;
    border-radius: 3px;
    color: transparent;
}

QProgressBar::chunk {
    background: #42bd91;
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
    background: #c6d6d0;
    min-height: 32px;
    border-radius: 3px;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QToolTip {
    background: #ffffff;
    color: #24332e;
    border: 1px solid #cadbd4;
    padding: 6px;
}
"""
