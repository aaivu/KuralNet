# KuralNet – Enhanced UI for Speech Emotion Recognition  
_Last Updated: 2025-04-11_

## Overview  
KuralNet’s enhanced UI provides a modern, responsive, and feature-rich frontend for visualizing speech emotion recognition (SER) results. Built with Tailwind CSS and JavaScript, it is designed for seamless integration with a SER backend.

---

## 📁 Files Included  
- `index.html` – Main HTML interface
- `styles.css` – Custom styling 
- `script.js` – Frontend logic

---

## ✨ Features

### 🎨 UI & Design
- Clean, modern layout using Tailwind CSS  
- Professional color gradients and shadows  
- Font Awesome icons  
- Fully responsive for mobile, tablet, and desktop  

### 📊 Emotion Analysis
- Tabs for **Current Segment** and **Overall Analysis**  
- Segment-wise emotion breakdowns  
- Timeframe-responsive waveform  
- Dynamic emotion flow chart  
- Real-time playback position tracking  

### 📤 Export Options
- Export emotion data as JSON or CSV  
- Save emotion charts as SVG images  

### 📱 Mobile Optimizations
- Touch-friendly waveform and controls  
- Smooth orientation handling  
- Optimized performance on mobile devices  

---

## 🔧 Integration Steps

To connect this UI to your SER backend:

1. **Update API Endpoint**  
   Edit the endpoint in `script.js` to point to your backend server.

2. **Handle Audio Uploads**  
   Your server should accept audio input and return emotion predictions.

3. **Return Standard Format**  
   Ensure your API responds with analysis data matching the UI's expected structure.

4. **Visualize Real Data**  
   Replace dummy values with live model predictions to activate dynamic charts and waveform sync.