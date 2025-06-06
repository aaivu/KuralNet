// KuralNet Speech Emotion Recognition UI JavaScript with API Integration

// API Configuration
const API_ENDPOINT = 'http://localhost:8787/predict';

// State variables
let segments = [];
let overallEmotions = {
    angry: 20,
    sad: 20,
    fear: 20,
    happy: 20,
    neutral: 20
};

// Map emotion to icon class
const emotionIcons = {
    angry: '<i class="fas fa-angry text-red-500"></i>',
    sad: '<i class="fas fa-sad-tear text-indigo-500"></i>',
    fear: '<i class="fas fa-surprise text-purple-500"></i>',
    happy: '<i class="fas fa-smile-beam text-green-500"></i>',
    neutral: '<i class="fas fa-meh text-gray-500"></i>'
};

// Map emotion to description
const emotionDescriptions = {
    angry: 'characterized by frustration or aggression',
    sad: 'showing signs of melancholy or sorrow',
    fear: 'exhibiting anxiety or apprehension',
    happy: 'expressing joy or positivity',
    neutral: 'a balanced emotional state'
};

// Emotion insights templates
const emotionInsights = {
    angry: 'This segment shows significant anger, which may indicate frustration or confrontation.',
    sad: 'The dominant emotion in this segment is sadness, suggesting disappointment or melancholy.',
    fear: 'Fear is prominently expressed in this segment, indicating anxiety or apprehension.',
    happy: 'This segment expresses happiness and positive emotions.',
    neutral: 'This segment maintains a predominantly neutral emotional tone.',
    mixed: 'This segment shows a mix of emotions, with %DOMINANT% being most prominent.'
};

// Audio duration variables
let audioDuration = 0;
let currentSegmentIndex = 0;
let audioBlob = null;

// Initialize the UI
document.addEventListener('DOMContentLoaded', function() {
    console.log('KuralNet UI initialized');
    
    // Set up navigation
    setupNavigation();
    
    // Set up tab switching
    setupTabs();
    
    // File upload button click
    const uploadBtn = document.getElementById('upload-btn');
    if (uploadBtn) {
        uploadBtn.addEventListener('click', () => {
            document.getElementById('audio-file').click();
        });
    }
    
    // File selected
    const audioFile = document.getElementById('audio-file');
    if (audioFile) {
        audioFile.addEventListener('change', handleFileUpload);
    }
    
    // Record button
    const recordBtn = document.getElementById('record-btn');
    if (recordBtn) {
        recordBtn.addEventListener('click', startRecording);
    }
    
    // Stop button
    const stopBtn = document.getElementById('stop-btn');
    if (stopBtn) {
        stopBtn.addEventListener('click', stopRecording);
    }
    
    // Play button
    const playBtn = document.getElementById('play-btn');
    if (playBtn) {
        playBtn.addEventListener('click', togglePlayback);
    }
    
    // Timeline slider
    const timelineSlider = document.getElementById('timeline-slider');
    if (timelineSlider) {
        timelineSlider.addEventListener('click', handleTimelineClick);
    }
    
    // Export buttons
    const exportJsonBtn = document.getElementById('export-json');
    if (exportJsonBtn) {
        exportJsonBtn.addEventListener('click', exportToJSON);
    }
    
    const exportCsvBtn = document.getElementById('export-csv');
    if (exportCsvBtn) {
        exportCsvBtn.addEventListener('click', exportToCSV);
    }
    
    const exportImageBtn = document.getElementById('export-image');
    if (exportImageBtn) {
        exportImageBtn.addEventListener('click', exportChart);
    }

    // Set up mobile touch events
    setupTouchEvents();
    
    // Adjust visualizations for mobile
    adjustVisualizationsForMobile();
});

// Navigation smooth scrolling
function setupNavigation() {
    const navLinks = document.querySelectorAll('nav a');
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            const targetId = e.target.getAttribute('href');
            if (targetId && targetId.startsWith('#')) {
                const targetSection = document.getElementById(targetId.slice(1));
                if (targetSection) {
                    e.preventDefault();
                    window.scrollTo({
                        top: targetSection.offsetTop - 80,
                        behavior: 'smooth'
                    });
                }
            }
        });
    });
}

// Tab switching
function setupTabs() {
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const tabId = tab.getAttribute('data-tab');
            if (!tabId) return;
            
            // Find parent container
            const tabContainer = tab.closest('div');
            if (!tabContainer) return;
            
            // Get all tabs in this container
            const siblingTabs = tabContainer.querySelectorAll('.tab');
            siblingTabs.forEach(t => {
                t.classList.remove('bg-white', 'text-blue-600', 'border-blue-500');
                t.classList.add('bg-gray-50', 'text-gray-700', 'border-transparent');
            });
            
            // Activate clicked tab
            tab.classList.remove('bg-gray-50', 'text-gray-700', 'border-transparent');
            tab.classList.add('bg-white', 'text-blue-600', 'border-blue-500');
            
            // Show tab content
            const tabContents = document.querySelectorAll('.tab-content');
            tabContents.forEach(content => {
                content.classList.add('hidden');
            });
            
            const activeContent = document.getElementById(tabId + '-tab');
            if (activeContent) {
                activeContent.classList.remove('hidden');
            }
        });
    });
}

// File upload handler
async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file) {
        // Validate file type
        const validTypes = ['audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/m4a', 'audio/ogg'];
        if (!validTypes.includes(file.type)) {
            showError('Please upload a valid audio file (WAV, MP3, M4A, or OGG)');
            return;
        }
        
        // Store the audio blob for playback
        audioBlob = file;
        
        // Show loading animation
        const loadingElement = document.querySelector('.loading');
        if (loadingElement) {
            loadingElement.classList.remove('hidden');
        }
        
        try {
            // Get audio duration
            audioDuration = await getAudioDuration(file);
            
            // Send file to API for emotion prediction
            const formData = new FormData();
            formData.append('file', file);
            
            // const response = await fetch(API_ENDPOINT, {
            //     method: 'POST',
            //     body: formData
            // });
            
            // if (!response.ok) {
            //     throw new Error(`API error: ${response.status}`);
            // }
            
            // const data = await response.json();
            // console.log('API response:', data);
            const data = [
                { start: 0, end: 15, mainEmotion: 'neutral', emotions: { angry: 15, sad: 30, fear: 10, happy: 5, neutral: 40 } },
                { start: 15, end: 30, mainEmotion: 'sad', emotions: { angry: 10, sad: 50, fear: 15, happy: 5, neutral: 20 } },
                { start: 30, end: 45, mainEmotion: 'angry', emotions: { angry: 60, sad: 10, fear: 15, happy: 5, neutral: 10 } },
                { start: 45, end: 75, mainEmotion: 'fear', emotions: { angry: 20, sad: 15, fear: 45, happy: 5, neutral: 15 } },
                { start: 75, end: 90, mainEmotion: 'happy', emotions: { angry: 5, sad: 5, fear: 5, happy: 75, neutral: 10 } },
                { start: 90, end: 120, mainEmotion: 'neutral', emotions: { angry: 5, sad: 10, fear: 5, happy: 10, neutral: 70 } }
            ];            
            // Process the API response

            processAPIResponse(data);
            
            // Hide loading and show visualization
            if (loadingElement) {
                loadingElement.classList.add('hidden');
            }
            
            const visualizationSection = document.getElementById('visualization-section');
            if (visualizationSection) {
                visualizationSection.classList.remove('hidden');
            }
            
            // Update button text with file name
            const uploadBtn = document.getElementById('upload-btn');
            if (uploadBtn) {
                uploadBtn.innerHTML = `<i class="fas fa-file-audio mr-2"></i>${file.name}`;
            }
            
            // Generate visualizations
            generateWaveform();
            generateTimeMarkers(audioDuration);
            updateTimeDisplay(0);
            
            if (segments.length > 0) {
                updateSegmentDetails(segments[0]);
            }
            
            generateEmotionFlowChart();
            generateDonutChart('overall-donut-chart', overallEmotions);
            
        } catch (error) {
            console.error('Error processing audio:', error);
            showError('Error processing audio file. Please try again.');
            
            if (loadingElement) {
                loadingElement.classList.add('hidden');
            }
        }
    }
}

// Get audio duration using Web Audio API
function getAudioDuration(file) {
    return new Promise((resolve, reject) => {
        const audio = new Audio();
        audio.addEventListener('loadedmetadata', () => {
            resolve(Math.floor(audio.duration));
        });
        audio.addEventListener('error', reject);
        audio.src = URL.createObjectURL(file);
    });
}

// Process API response and convert to segments format
function processAPIResponse(apiData) {
    // Clear existing segments
    segments = [];
    
    // Assuming the API returns data in a similar format to your dummy data
    // Adjust this based on the actual API response structure
    if (apiData.segments) {
        segments = apiData.segments.map(segment => ({
            start: segment.start,
            end: segment.end,
            mainEmotion: segment.mainEmotion || getDominantEmotion(segment.emotions),
            emotions: segment.emotions
        }));
    }
    
    // Calculate overall emotions if not provided
    if (apiData.overallEmotions) {
        overallEmotions = apiData.overallEmotions;
    } else {
        calculateOverallEmotions();
    }
}

// Get dominant emotion from emotion values
function getDominantEmotion(emotions) {
    let maxEmotion = 'neutral';
    let maxValue = 0;
    
    for (const [emotion, value] of Object.entries(emotions)) {
        if (value > maxValue) {
            maxValue = value;
            maxEmotion = emotion;
        }
    }
    
    return maxEmotion;
}

// Calculate overall emotions from segments
function calculateOverallEmotions() {
    // Reset overall emotions
    for (const emotion in overallEmotions) {
        overallEmotions[emotion] = 0;
    }
    
    // Calculate total duration
    const totalDuration = segments.reduce((total, segment) => total + (segment.end - segment.start), 0);
    
    // Calculate weighted average of emotions
    segments.forEach(segment => {
        const segmentDuration = segment.end - segment.start;
        const weight = segmentDuration / totalDuration;
        
        for (const emotion in segment.emotions) {
            overallEmotions[emotion] = (overallEmotions[emotion] || 0) + (segment.emotions[emotion] * weight);
        }
    });
    
    // Round values
    for (const emotion in overallEmotions) {
        overallEmotions[emotion] = Math.round(overallEmotions[emotion]);
    }
    
    // Normalize to ensure sum equals 100
    let sum = Object.values(overallEmotions).reduce((total, val) => total + val, 0);
    if (sum !== 100) {
        const scaleFactor = 100 / sum;
        for (const emotion in overallEmotions) {
            overallEmotions[emotion] = Math.round(overallEmotions[emotion] * scaleFactor);
        }
        
        // Adjust rounding errors
        sum = Object.values(overallEmotions).reduce((total, val) => total + val, 0);
        if (sum !== 100) {
            const diff = 100 - sum;
            // Add or subtract the difference from the largest emotion
            let largestEmotion = Object.keys(overallEmotions).reduce((a, b) => 
                overallEmotions[a] > overallEmotions[b] ? a : b);
            overallEmotions[largestEmotion] += diff;
        }
    }
}

// Recording functions
let mediaRecorder;
let recordedChunks = [];

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        recordedChunks = [];
        
        mediaRecorder.ondataavailable = event => {
            if (event.data.size > 0) {
                recordedChunks.push(event.data);
            }
        };
        
        mediaRecorder.onstop = async () => {
            const recordedBlob = new Blob(recordedChunks, { type: 'audio/webm' });
            audioBlob = recordedBlob;
            
            // Process the recorded audio
            await processRecordedAudio(recordedBlob);
        };
        
        mediaRecorder.start();
        
        // Update UI
        document.getElementById('record-btn').classList.add('hidden');
        document.getElementById('stop-btn').classList.remove('hidden');
        document.getElementById('rec-indicator').classList.remove('hidden');
        
        // Update record box style
        const recordBox = document.querySelector('.record-box');
        if (recordBox) {
            recordBox.classList.add('border-red-400', 'bg-red-50');
        }
    } catch (error) {
        console.error('Error starting recording:', error);
        showError('Unable to access microphone. Please check permissions.');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
    }
    
    // Update UI
    document.getElementById('stop-btn').classList.add('hidden');
    document.getElementById('record-btn').classList.remove('hidden');
    document.getElementById('record-btn').innerHTML = '<i class="fas fa-microphone mr-2"></i> Record Again';
    document.getElementById('rec-indicator').classList.add('hidden');
    
    // Reset record box style
    const recordBox = document.querySelector('.record-box');
    if (recordBox) {
        recordBox.classList.remove('border-red-400', 'bg-red-50');
    }
}

async function processRecordedAudio(blob) {
    // Show loading
    const loadingElement = document.querySelector('.loading');
    if (loadingElement) {
        loadingElement.classList.remove('hidden');
    }
    
    try {
        // Get audio duration
        audioDuration = await getAudioDuration(blob);
        
        // Send to API
        const formData = new FormData();
        formData.append('file', blob, 'recording.webm');
        
        // const response = await fetch(API_ENDPOINT, {
        //     method: 'POST',
        //     body: formData
        // });
        
        // if (!response.ok) {
        //     throw new Error(`API error: ${response.status}`);
        // }
        
        // const data = await response.json();

        const data = [
            { start: 0, end: 15, mainEmotion: 'neutral', emotions: { angry: 15, sad: 30, fear: 10, happy: 5, neutral: 40 } },
            { start: 15, end: 30, mainEmotion: 'sad', emotions: { angry: 10, sad: 50, fear: 15, happy: 5, neutral: 20 } },
            { start: 30, end: 45, mainEmotion: 'angry', emotions: { angry: 60, sad: 10, fear: 15, happy: 5, neutral: 10 } },
            { start: 45, end: 75, mainEmotion: 'fear', emotions: { angry: 20, sad: 15, fear: 45, happy: 5, neutral: 15 } },
            { start: 75, end: 90, mainEmotion: 'happy', emotions: { angry: 5, sad: 5, fear: 5, happy: 75, neutral: 10 } },
            { start: 90, end: 120, mainEmotion: 'neutral', emotions: { angry: 5, sad: 10, fear: 5, happy: 10, neutral: 70 } }
        ];
        
        // Process the API response
        processAPIResponse(data);
        
        // Hide loading and show visualization
        if (loadingElement) {
            loadingElement.classList.add('hidden');
        }
        
        const visualizationSection = document.getElementById('visualization-section');
        if (visualizationSection) {
            visualizationSection.classList.remove('hidden');
        }
        
        // Generate visualizations
        generateWaveform();
        generateTimeMarkers(audioDuration);
        updateTimeDisplay(0);
        
        if (segments.length > 0) {
            updateSegmentDetails(segments[0]);
        }
        
        generateEmotionFlowChart();
        generateDonutChart('overall-donut-chart', overallEmotions);
        
    } catch (error) {
        console.error('Error processing recording:', error);
        showError('Error processing recording. Please try again.');
        
        if (loadingElement) {
            loadingElement.classList.add('hidden');
        }
    }
}

// Show error message
function showError(message) {
    // You can implement a toast notification or alert here
    alert(message);
}

// Generate dynamic time markers based on audio duration
function generateTimeMarkers(duration) {
    // Determine appropriate time intervals based on duration
    let interval;
    if (duration <= 60) {
        interval = 10; // 10-second intervals for shorter audio
    } else if (duration <= 180) {
        interval = 15; // 15-second intervals for medium audio
    } else if (duration <= 300) {
        interval = 30; // 30-second intervals for longer audio
    } else {
        interval = 60; // 1-minute intervals for very long audio
    }
    
    // Generate time labels
    const timeLabels = document.getElementById('time-labels');
    if (timeLabels) {
        timeLabels.innerHTML = '';
        
        // Calculate number of labels
        const numLabels = Math.ceil(duration / interval) + 1;
        
        // Create labels with flex justification
        for (let i = 0; i < numLabels; i++) {
            const time = i * interval;
            if (time <= duration) {
                const label = document.createElement('span');
                label.textContent = formatTime(time);
                timeLabels.appendChild(label);
            }
        }
    }
    
    // Generate time markers in waveform
    const timeMarkers = document.getElementById('time-markers');
    if (timeMarkers) {
        timeMarkers.innerHTML = '';
        
        // Create markers at each interval
        for (let time = 0; time <= duration; time += interval) {
            const position = (time / duration) * 100;
            
            const marker = document.createElement('div');
            marker.className = 'time-marker';
            marker.style.left = `${position}%`;
            
            const label = document.createElement('div');
            label.className = 'time-marker-label';
            label.textContent = formatTime(time);
            
            marker.appendChild(label);
            timeMarkers.appendChild(marker);
        }
    }
}

// Generate waveform with dynamic segments based on audio duration
function generateWaveform() {
    const waveform = document.getElementById('waveform');
    if (!waveform) return;
    
    // Clear existing content
    waveform.innerHTML = '';
    
    // Create waveform lines background
    const waveformLines = document.createElement('div');
    waveformLines.className = 'flex items-center h-full w-full absolute';
    
    // Create random lines for waveform visualization
    const numLines = Math.floor(waveform.offsetWidth / 3); // Approximate 3px per line
    for (let i = 0; i < numLines; i++) {
        const line = document.createElement('div');
        const height = 30 + Math.random() * 70; // 30% to 100% height
        
        line.className = 'flex-1 h-[' + height + '%] bg-gray-300 mx-px rounded-px waveform-line';
        line.style.animationDelay = `${Math.random() * 1}s`;
        line.style.height = `${height}%`;
        waveformLines.appendChild(line);
    }
    
    waveform.appendChild(waveformLines);
    
    // Add segments based on API data
    segments.forEach((segment, index) => {
        const segmentElement = document.createElement('div');
        segmentElement.className = 'absolute h-full opacity-80 hover:opacity-100 cursor-pointer transition-opacity duration-300';
        
        // Calculate position and width based on time
        const startPercent = (segment.start / audioDuration) * 100;
        const endPercent = (segment.end / audioDuration) * 100;
        const width = endPercent - startPercent;
        
        segmentElement.style.left = `${startPercent}%`;
        segmentElement.style.width = `${width}%`;
        segmentElement.classList.add(`emotion-${segment.mainEmotion}`);
        
        // Create tooltip
        const tooltip = document.createElement('div');
        tooltip.className = 'absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-2 bg-gray-900 text-white text-xs rounded whitespace-nowrap hidden group-hover:block z-20';
        tooltip.textContent = `${formatTime(segment.start)} - ${formatTime(segment.end)}: ${capitalizeFirstLetter(segment.mainEmotion)}`;
        segmentElement.appendChild(tooltip);
        segmentElement.classList.add('group');
        
        // Add click event to show segment details
        segmentElement.addEventListener('click', () => {
            currentSegmentIndex = index;
            updateSegmentDetails(segment);
            
            // Switch to segment tab
            const segmentTab = document.querySelector('[data-tab="segment"]');
            if (segmentTab) {
                segmentTab.click();
            }
            
            // Highlight the clicked segment
            highlightSegment(segmentElement);
        });
        
        waveform.appendChild(segmentElement);
    });
    
    // Set initial segment details
    if (segments.length > 0) {
        updateSegmentDetails(segments[0]);
    }
}

// Highlight the selected segment
function highlightSegment(segmentElement) {
    // Remove highlight from all segments
    const allSegments = document.querySelectorAll('#waveform > div.absolute.h-full');
    allSegments.forEach(seg => {
        seg.classList.remove('ring-2', 'ring-white', 'ring-opacity-70', 'z-10');
    });
    
    // Add highlight to selected segment
    segmentElement.classList.add('ring-2', 'ring-white', 'ring-opacity-70', 'z-10');
}

// Playback controls
let isPlaying = false;
let playbackInterval;
let currentPlaybackPercent = 0;
let audioElement = null;

function togglePlayback() {
    const playBtn = document.getElementById('play-btn');
    
    if (!audioBlob) {
        showError('No audio loaded');
        return;
    }
    
    if (isPlaying) {
        // Pause playback
        isPlaying = false;
        playBtn.innerHTML = '<i class="fas fa-play"></i>';
        clearInterval(playbackInterval);
        
        if (audioElement) {
            audioElement.pause();
        }
    } else {
        // Start playback
        isPlaying = true;
        playBtn.innerHTML = '<i class="fas fa-pause"></i>';
        
        // Create audio element if not exists
        if (!audioElement) {
            audioElement = new Audio();
            audioElement.src = URL.createObjectURL(audioBlob);
            
            audioElement.addEventListener('ended', () => {
                isPlaying = false;
                playBtn.innerHTML = '<i class="fas fa-play"></i>';
                currentPlaybackPercent = 0;
                updatePlaybackPosition(0);
            });
        }
        
        audioElement.play();
        
        // Update progress
        playbackInterval = setInterval(() => {
            if (audioElement && audioElement.duration) {
                currentPlaybackPercent = audioElement.currentTime / audioElement.duration;
                updatePlaybackPosition(currentPlaybackPercent);
            }
        }, 100);
    }
}

function updatePlaybackPosition(percent) {
    const timelineHandle = document.getElementById('timeline-handle');
    const timelineProgress = document.getElementById('timeline-progress');
    
    if (timelineHandle) timelineHandle.style.left = `${percent * 100}%`;
    if (timelineProgress) timelineProgress.style.width = `${percent * 100}%`;
    
    updateTimeDisplay(percent);
    updateFlowChartPosition(percent);
    
    // Find current segment based on time
    const currentTime = Math.floor(percent * audioDuration);
    
    let currentSegment = null;
    let segmentIndex = 0;
    
    for (let i = 0; i < segments.length; i++) {
        const segment = segments[i];
        if (currentTime >= segment.start && currentTime < segment.end) {
            currentSegment = segment;
            segmentIndex = i;
            break;
        }
    }
    
    if (currentSegment && segmentIndex !== currentSegmentIndex) {
        currentSegmentIndex = segmentIndex;
        updateSegmentDetails(currentSegment);
        
        // Highlight the current segment in the waveform
        const segmentElements = document.querySelectorAll('#waveform > div.absolute.h-full');
        if (segmentElements[segmentIndex]) {
            highlightSegment(segmentElements[segmentIndex]);
        }
    }
}

function handleTimelineClick(e) {
    const timelineSlider = document.getElementById('timeline-slider');
    if (!timelineSlider) return;
    
    const rect = timelineSlider.getBoundingClientRect();
    const percent = Math.min(Math.max(0, (e.clientX - rect.left) / rect.width), 1);
    
    // Update playback position
    currentPlaybackPercent = percent;
    updatePlaybackPosition(percent);
    
    // Seek audio if available
    if (audioElement && audioElement.duration) {
        audioElement.currentTime = percent * audioElement.duration;
    }
}

function updateTimeDisplay(percent) {
    const currentTime = Math.floor(percent * audioDuration);
    
    const currentMinutes = Math.floor(currentTime / 60);
    const currentSeconds = currentTime % 60;
    const totalMinutes = Math.floor(audioDuration / 60);
    const totalSeconds = audioDuration % 60;
    
    const timeDisplay = document.getElementById('time-display');
    if (timeDisplay) {
        timeDisplay.textContent = `${currentMinutes}:${currentSeconds < 10 ? '0' : ''}${currentSeconds} / ${totalMinutes}:${totalSeconds < 10 ? '0' : ''}${totalSeconds}`;
    }
}

// Update segment details when clicked
function updateSegmentDetails(segment) {
    // Update current segment time display
    const currentSegmentTime = document.getElementById('current-segment-time');
    if (currentSegmentTime) {
        currentSegmentTime.textContent = `${formatTime(segment.start)} - ${formatTime(segment.end)}`;
    }
    
    // Update segment dominant emotion
    const segmentDominant = document.getElementById('segment-dominant');
    if (segmentDominant) {
        segmentDominant.textContent = capitalizeFirstLetter(segment.mainEmotion);
    }
    
    // Update segment donut chart
    console.log("766:" , segment.emotions);
    generateDonutChart('segment-donut-chart', segment.emotions);
    
    // Update segment insights
    updateSegmentInsights(segment);
    
    // Update emotion percentages
    const emotions = ['angry', 'sad', 'fear', 'happy', 'neutral'];
    
    emotions.forEach(emotion => {
        const progressBar = document.querySelector(`#segment-tab .${emotion}-progress`);
        const percentage = document.querySelector(`#segment-tab .${emotion}-percentage`);
        
        if (progressBar) progressBar.style.width = `${segment.emotions[emotion]}%`;
        if (percentage) percentage.textContent = `${segment.emotions[emotion]}%`;
    });
    
    // Update dominant emotion icon and text
    const dominantIcon = document.getElementById('dominant-icon');
    const dominantEmotionName = document.getElementById('dominant-emotion-name');
    const dominantEmotionDesc = document.getElementById('dominant-emotion-desc');
    
    if (dominantIcon) dominantIcon.innerHTML = emotionIcons[segment.mainEmotion];
    if (dominantEmotionName) dominantEmotionName.textContent = capitalizeFirstLetter(segment.mainEmotion);
    if (dominantEmotionDesc) dominantEmotionDesc.textContent = emotionDescriptions[segment.mainEmotion];
}

// Update segment insights based on emotion distribution
function updateSegmentInsights(segment) {
    const segmentInsights = document.getElementById('segment-insights');
    if (!segmentInsights) return;
    
    // Get dominant and secondary emotions
    const emotions = Object.entries(segment.emotions).sort((a, b) => b[1] - a[1]);
    const dominant = emotions[0];
    const secondary = emotions[1];
    
    // Check if there's a clear dominant emotion (more than 50%)
    if (dominant[1] > 50) {
        segmentInsights.textContent = emotionInsights[dominant[0]];
    } else {
        // Mixed emotions
        let insight = emotionInsights.mixed.replace('%DOMINANT%', capitalizeFirstLetter(dominant[0]));
        insight += ` There is also a significant presence of ${secondary[0]} (${secondary[1]}%).`;
        segmentInsights.textContent = insight;
    }
    
    // Add additional context based on specific combinations
    if (dominant[0] === 'angry' && segment.emotions.sad > 20) {
        segmentInsights.textContent += ' The combination of anger and sadness may indicate frustration or resentment.';
    } else if (dominant[0] === 'sad' && segment.emotions.fear > 20) {
        segmentInsights.textContent += ' The mixture of sadness and fear suggests anxiety or despair.';
    } else if (dominant[0] === 'happy' && segment.emotions.neutral > 30) {
        segmentInsights.textContent += ' The balance of happiness and neutrality indicates a calm, positive state.';
    }
}

// Generate donut chart for emotion visualization
function generateDonutChart(containerId, emotions) {
    console.log('Generating donut chart for:', emotions);
    const container = document.getElementById(containerId);
    if (!container) return;
    
    const centerDiv = container.querySelector('.donut-center');
    
    // Remove existing SVG
    const existingSvg = container.querySelector('svg');
    if (existingSvg) {
        container.removeChild(existingSvg);
    }
    
    // Create SVG
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', '100%');
    svg.setAttribute('viewBox', '0 0 42 42');
    
    // Calculate total for percentages
    let total = Object.values(emotions).reduce((acc, val) => acc + (isNaN(val) ? 0 : val), 0);
    // If total is 0, set it to 100 to avoid division by zero
    if (total === 0) {
        total = 100;
    }
    
    // Skip if total is 0 to avoid division by zero
    if (total <= 0) {
        console.warn('Cannot generate donut chart: total emotion values sum to 0');
        return;
    }
    
    // Track current angle for drawing segments
    let currentAngle = 0;
    
    // Get colors based on emotion
    const getEmotionColor = (emotion) => {
        const colorMap = {
            'angry': '#ef4444',  // red-500
            'sad': '#6366f1',    // indigo-500
            'fear': '#a855f7',   // purple-500
            'happy': '#22c55e',  // green-500
            'neutral': '#6b7280' // gray-500
        };
        return colorMap[emotion] || '#6b7280';
    };
    
    // Draw donut chart segments
    for (const emotion in emotions) {
        if (emotions[emotion] === 0) continue; // Skip zero values
        
        const percentage = emotions[emotion] / total;
        const angle = percentage * 360;
        
        // Calculate start and end points
        const startAngle = currentAngle;
        const endAngle = currentAngle + angle;
        // Convert to radians for Math.cos and Math.sin
        const startRad = startAngle * (Math.PI / 180);
        const endRad = endAngle * (Math.PI / 180);
        // Calculate path coordinates
        const x1 = 21 + 15 * Math.cos(startRad);
        const y1 = 21 + 15 * Math.sin(startRad);
        const x2 = 21 + 15 * Math.cos(endRad);
        const y2 = 21 + 15 * Math.sin(endRad);
        
        // Skip invalid coordinates
        if (isNaN(x1) || isNaN(y1) || isNaN(x2) || isNaN(y2)) {
            console.warn(`Skipping invalid segment with coordinates: x1=${x1}, y1=${y1}, x2=${x2}, y2=${y2}`);
            continue;
        }
        
        // Determine if the arc should be drawn as a large arc
        const largeArcFlag = angle > 180 ? 1 : 0;
        
        // Create path
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d', `M 21 21 L ${x1} ${y1} A 15 15 0 ${largeArcFlag} 1 ${x2} ${y2} Z`);
        path.setAttribute('fill', getEmotionColor(emotion));
        path.setAttribute('class', 'donut-segment');
        
        // Add title for tooltip
        const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
        title.textContent = `${capitalizeFirstLetter(emotion)}: ${emotions[emotion]}%`;
        path.appendChild(title);
        
        svg.appendChild(path);
        
        // Update current angle
        currentAngle += angle;
    }
    
    // Insert SVG before center element
    if (centerDiv) {
        container.insertBefore(svg, centerDiv);
    } else {
        container.appendChild(svg);
    }
}

// Generate emotion flow chart
function generateEmotionFlowChart() {
    const flowChartContainer = document.getElementById('emotion-flow-chart');
    if (!flowChartContainer) return;
    
    // Clear existing content
    flowChartContainer.innerHTML = '';
    
    // Create SVG for the chart
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', '100%');
    svg.setAttribute('viewBox', '0 0 1000 300');
    svg.setAttribute('class', 'overflow-visible');
    
    // Create a group for grid lines
    const gridGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    gridGroup.setAttribute('stroke', '#e5e7eb');
    gridGroup.setAttribute('stroke-width', '1');
    
    // Add horizontal grid lines
    for (let i = 0; i <= 5; i++) {
        const y = 300 - (i * 60);
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', '0');
        line.setAttribute('y1', y);
        line.setAttribute('x2', '1000');
        line.setAttribute('y2', y);
        gridGroup.appendChild(line);
        
        // Add y-axis labels
        if (i > 0) {
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', '-5');
            text.setAttribute('y', y + 5);
            text.setAttribute('text-anchor', 'end');
            text.setAttribute('font-size', '12');
            text.setAttribute('fill', '#6b7280');
            text.textContent = `${i * 20}%`;
            gridGroup.appendChild(text);
        }
    }
    
    svg.appendChild(gridGroup);
    
    // Create groups for each emotion line
    const emotions = ['angry', 'sad', 'fear', 'happy', 'neutral'];
    const colorMap = {
        'angry': '#ef4444',  // red-500
        'sad': '#6366f1',    // indigo-500
        'fear': '#a855f7',   // purple-500
        'happy': '#22c55e',  // green-500
        'neutral': '#6b7280' // gray-500
    };
    
    // Calculate x-axis spacing based on segments
    const totalDuration = audioDuration;
    const xScale = 1000 / totalDuration;
    
    // Draw emotion lines
    emotions.forEach(emotion => {
        // Create path for this emotion
        const pathData = [];
        
        segments.forEach((segment, index) => {
            const segmentStart = segment.start;
            const segmentEnd = segment.end;
            const emotionValue = segment.emotions[emotion] || 0;
            
            // Calculate coordinates
            const x1 = segmentStart * xScale;
            const y1 = 300 - (emotionValue * 3); // Scale: 100% would be at y=0, 0% at y=300
            
            if (index === 0) {
                pathData.push(`M ${x1} ${y1}`);
            } else {
                pathData.push(`L ${x1} ${y1}`);
            }
            
            // Add points at segment end
            const x2 = segmentEnd * xScale;
            pathData.push(`L ${x2} ${y1}`);
        });
        
        // Create the path element
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d', pathData.join(' '));
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke', colorMap[emotion]);
        path.setAttribute('stroke-width', '3');
        path.setAttribute('stroke-linejoin', 'round');
        path.setAttribute('stroke-linecap', 'round');
        svg.appendChild(path);
        
        // Add emotion label at the end
        if (segments.length > 0) {
            const lastSegment = segments[segments.length - 1];
            const lastX = lastSegment.end * xScale + 10;
            const lastY = 300 - ((lastSegment.emotions[emotion] || 0) * 3);
            
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', lastX);
            text.setAttribute('y', lastY + 5);
            text.setAttribute('font-size', '12');
            text.setAttribute('fill', colorMap[emotion]);
            text.textContent = capitalizeFirstLetter(emotion);
            svg.appendChild(text);
        }
    });
    
    // Add time markers on x-axis
    const timeGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    
    // Determine appropriate interval for x-axis
    let interval;
    if (totalDuration <= 60) {
        interval = 10; // 10 seconds
    } else if (totalDuration <= 180) {
        interval = 30; // 30 seconds
    } else {
        interval = 60; // 1 minute
    }
    
    for (let time = 0; time <= totalDuration; time += interval) {
        const x = time * xScale;
        
        // Add vertical line
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', x);
        line.setAttribute('y1', '300');
        line.setAttribute('x2', x);
        line.setAttribute('y2', '305');
        line.setAttribute('stroke', '#6b7280');
        line.setAttribute('stroke-width', '1');
        timeGroup.appendChild(line);
        
        // Add time label
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', x);
        text.setAttribute('y', '320');
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('font-size', '12');
        text.setAttribute('fill', '#6b7280');
        text.textContent = formatTime(time);
        timeGroup.appendChild(text);
    }
    
    svg.appendChild(timeGroup);
    
    // Add a playback position indicator line
    const positionLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    positionLine.setAttribute('id', 'flow-chart-position');
    positionLine.setAttribute('x1', '0');
    positionLine.setAttribute('y1', '0');
    positionLine.setAttribute('x2', '0');
    positionLine.setAttribute('y2', '300');
    positionLine.setAttribute('stroke', '#3b82f6');
    positionLine.setAttribute('stroke-width', '2');
    positionLine.setAttribute('stroke-dasharray', '5,5');
    positionLine.style.display = 'none';
    svg.appendChild(positionLine);
    
    flowChartContainer.appendChild(svg);
}

// Update flow chart position indicator during playback
function updateFlowChartPosition(percent) {
    const positionLine = document.getElementById('flow-chart-position');
    if (!positionLine) return;
    
    const x = percent * 1000; // 1000 is the SVG viewBox width
    positionLine.setAttribute('x1', x);
    positionLine.setAttribute('x2', x);
    positionLine.style.display = 'block';
}

function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds < 10 ? '0' : ''}${remainingSeconds}`;
}

function capitalizeFirstLetter(string) {
    return string.charAt(0).toUpperCase() + string.slice(1);
}

// Export functionality
function exportToJSON() {
    // Create a data object with all analysis results
    const analysisData = {
        fileName: document.getElementById('upload-btn').textContent || 'recorded_audio',
        duration: audioDuration,
        timestamp: new Date().toISOString(),
        dominantEmotion: getOverallDominantEmotion(),
        segments: segments,
        overallEmotions: overallEmotions
    };
    
    // Convert to JSON string
    const jsonString = JSON.stringify(analysisData, null, 2);
    
    // Create blob and download link
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    // Create download link
    const a = document.createElement('a');
    a.href = url;
    a.download = 'emotion_analysis.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    
    // Clean up
    URL.revokeObjectURL(url);
    
    // Show status
    showExportStatus('JSON file downloaded successfully!');
}

function exportToCSV() {
    // Create CSV header
    let csvContent = 'Segment Start,Segment End,Dominant Emotion,Angry %,Sad %,Fear %,Happy %,Neutral %\n';
    
    // Add segment data
    segments.forEach(segment => {
        csvContent += `${formatTime(segment.start)},${formatTime(segment.end)},${segment.mainEmotion},` + 
                      `${segment.emotions.angry || 0},${segment.emotions.sad || 0},${segment.emotions.fear || 0},` +
                      `${segment.emotions.happy || 0},${segment.emotions.neutral || 0}\n`;
    });
    
    // Add overall data
    csvContent += `\nOverall,${formatTime(audioDuration)},${getOverallDominantEmotion()},` +
                  `${overallEmotions.angry},${overallEmotions.sad},${overallEmotions.fear},` +
                  `${overallEmotions.happy},${overallEmotions.neutral}\n`;
    
    // Create blob and download link
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    
    // Create download link
    const a = document.createElement('a');
    a.href = url;
    a.download = 'emotion_analysis.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    
    // Clean up
    URL.revokeObjectURL(url);
    
    // Show status
    showExportStatus('CSV file downloaded successfully!');
}

function exportChart() {
    const flowChartContainer = document.getElementById('emotion-flow-chart');
    if (!flowChartContainer) {
        showExportStatus('Chart not available!', true);
        return;
    }
    
    // Get the SVG element
    const svg = flowChartContainer.querySelector('svg');
    if (!svg) {
        showExportStatus('Chart not available!', true);
        return;
    }
    
    // Create a serialized SVG string
    const serializer = new XMLSerializer();
    let svgString = serializer.serializeToString(svg);
    
    // Fix SVG namespace issues if any
    svgString = svgString.replace(/^<svg/, '<svg xmlns="http://www.w3.org/2000/svg"');
    
    // Add XML declaration
    svgString = '<?xml version="1.0" standalone="no"?>\r\n' + svgString;
    
    // Convert SVG string to a blob
    const blob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    
    // Create download link
    const a = document.createElement('a');
    a.href = url;
    a.download = 'emotion_flow_chart.svg';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    
    // Clean up
    URL.revokeObjectURL(url);
    
    // Show status
    showExportStatus('Chart exported as SVG successfully!');
}

function showExportStatus(message, isError = false) {
    const statusElement = document.getElementById('export-status');
    const messageElement = document.getElementById('export-message');
    
    if (statusElement && messageElement) {
        messageElement.textContent = message;
        statusElement.classList.remove('hidden');
        
        if (isError) {
            statusElement.querySelector('i').className = 'fas fa-exclamation-circle text-red-500 mr-1';
        } else {
            statusElement.querySelector('i').className = 'fas fa-check-circle text-green-500 mr-1';
        }
        
        // Hide after 3 seconds
        setTimeout(() => {
            statusElement.classList.add('hidden');
        }, 3000);
    }
}

function getOverallDominantEmotion() {
    let dominantEmotion = 'neutral';
    let maxValue = 0;
    
    for (const emotion in overallEmotions) {
        if (overallEmotions[emotion] > maxValue) {
            maxValue = overallEmotions[emotion];
            dominantEmotion = emotion;
        }
    }
    
    return dominantEmotion;
}

// Touch event handling for mobile devices
function setupTouchEvents() {
    const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
    if (!isTouchDevice) return;
    
    console.log('Setting up touch events for mobile');
    
    // Add touch-feedback class to buttons
    document.querySelectorAll('button').forEach(button => {
        button.classList.add('touch-feedback');
    });
    
    // Handle touch events for timeline slider
    const timelineSlider = document.getElementById('timeline-slider');
    if (timelineSlider) {
        // Remove click handler to prevent duplicate events
        const oldClickHandler = timelineSlider.onclick;
        timelineSlider.onclick = null;
        
        // Touch start handler
        timelineSlider.addEventListener('touchstart', handleTimelineTouch, { passive: false });
        timelineSlider.addEventListener('touchmove', handleTimelineTouch, { passive: false });
        
        // Restore click handler for non-touch devices
        if (oldClickHandler) {
            timelineSlider.addEventListener('click', oldClickHandler);
        }
    }
    
    // Handle touch events for waveform segments
    const waveform = document.getElementById('waveform');
    if (waveform) {
        waveform.querySelectorAll('div.absolute.h-full').forEach(segment => {
            segment.classList.add('touch-feedback');
        });
    }
    
    // Add touch feedback to export buttons
    ['export-json', 'export-csv', 'export-image'].forEach(id => {
        const btn = document.getElementById(id);
        if (btn) btn.classList.add('touch-feedback');
    });
}

// Handle touch events on timeline
function handleTimelineTouch(e) {
    e.preventDefault();
    
    const timelineSlider = document.getElementById('timeline-slider');
    if (!timelineSlider) return;
    
    const touch = e.touches[0];
    const rect = timelineSlider.getBoundingClientRect();
    let percent = Math.min(Math.max(0, (touch.clientX - rect.left) / rect.width), 1);
    
    // Update playback position
    currentPlaybackPercent = percent;
    updatePlaybackPosition(percent);
    
    // Seek audio if available
    if (audioElement && audioElement.duration) {
        audioElement.currentTime = percent * audioElement.duration;
    }
}

// Adjust visualizations for mobile
function adjustVisualizationsForMobile() {
    const isMobile = window.innerWidth < 768;
    
    if (isMobile) {
        // Simplify waveform on mobile
        const waveform = document.getElementById('waveform');
        if (waveform) {
            const waveformLines = waveform.querySelector('.flex.items-center.h-full.w-full.absolute');
            if (waveformLines) {
                // Keep only half the lines for better performance
                const lines = waveformLines.querySelectorAll('div');
                for (let i = 0; i < lines.length; i++) {
                    if (i % 2 !== 0) {
                        lines[i].style.display = 'none';
                    }
                }
            }
        }
        
        // Adjust time markers
        const timeLabels = document.getElementById('time-labels');
        if (timeLabels) {
            const spans = timeLabels.querySelectorAll('span');
            // Hide every other label on mobile
            for (let i = 0; i < spans.length; i++) {
                if (i % 2 !== 0 && i !== spans.length - 1) {
                    spans[i].style.display = 'none';
                }
            }
        }
    }
}

// Detect orientation change
function handleOrientationChange() {
    // Re-adjust visualizations when orientation changes
    adjustVisualizationsForMobile();
    
    // Re-generate charts for new dimensions
    if (document.getElementById('segment-donut-chart') && segments.length > 0) {
        console.log('1353 :', segments[currentSegmentIndex].emotions);
        generateDonutChart('segment-donut-chart', segments[currentSegmentIndex].emotions);
    }
    
    if (document.getElementById('overall-donut-chart')) {
        console.log('1358 :', overallEmotions);
        generateDonutChart('overall-donut-chart', overallEmotions);
    }
    
    if (document.getElementById('emotion-flow-chart')) {
        generateEmotionFlowChart();
    }
}

// Setup orientation change event
window.addEventListener('orientationchange', function() {
    // Small delay to ensure dimensions have updated
    setTimeout(handleOrientationChange, 300);
});

// Also listen for resize events (for desktop/browser testing)
let resizeTimeout;
window.addEventListener('resize', function() {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(handleOrientationChange, 300);
});

// Show mobile orientation message
function showMobileMessage() {
    const isMobile = window.innerWidth < 768;
    const isPortrait = window.innerHeight > window.innerWidth;
    const mobileMessage = document.getElementById('mobile-message');
    
    if (mobileMessage) {
        if (isMobile && isPortrait) {
            mobileMessage.classList.remove('hidden');
        } else {
            mobileMessage.classList.add('hidden');
        }
    }
}

// Check orientation on load and changes
window.addEventListener('DOMContentLoaded', showMobileMessage);
window.addEventListener('orientationchange', showMobileMessage);
window.addEventListener('resize', showMobileMessage);