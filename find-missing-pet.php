<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Find Missing Pet - Fur-Get Me Not</title>
    <link rel="stylesheet" href="css/style.css">
</head>
<body style="background: url('assets/Home/bg-rectangle.png') center/cover no-repeat fixed;">
    <div class="bg-paws"></div>
    <!-- Loading overlay -->
    <div id="find-loading-overlay" style="display:none;position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:9999;background:rgba(248,250,255,0.85);backdrop-filter:blur(2px);align-items:center;justify-content:center;flex-direction:column;">
        <div style="display:flex;flex-direction:column;align-items:center;max-width:520px;">
            <div class="find-spinner" style="width:64px;height:64px;margin-bottom:18px;">
                <svg viewBox="0 0 50 50" style="width:100%;height:100%;">
                    <circle cx="25" cy="25" r="20" fill="none" stroke="#3867d6" stroke-width="6" stroke-linecap="round" stroke-dasharray="90 60" stroke-dashoffset="0">
                        <animateTransform attributeName="transform" type="rotate" from="0 25 25" to="360 25 25" dur="1s" repeatCount="indefinite"/>
                    </circle>
                </svg>
            </div>
            <div id="find-progress-stage" role="status" aria-live="polite" style="font-size:1.1rem;color:#223a7b;font-weight:700;margin-bottom:14px;text-align:center;min-height:26px;display:flex;align-items:center;justify-content:center;">Starting...</div>
            <div id="find-progress-detail" aria-live="polite" style="min-height:48px;display:flex;align-items:center;justify-content:center;padding:0 6px;font-size:0.9rem;color:#3867d6;text-align:center;line-height:1.35;">Preparing preprocessing pipeline...</div>
        </div>
    </div>
    <header>
        <div class="how-header-bar">
            <img src="assets/Home/Rectangle 6.png" alt="Header Rectangle" class="header-rectangle">
            <a href="index.php" class="how-back-arrow" aria-label="Back to Home">
                <img src="assets/How-it-Works/back-arrow.png" alt="Back Arrow" style="width:36px;height:36px;">
            </a>
        </div>
        <div class="find-header">
            <img src="assets/Logos/pawprint-blue 1.png" alt="Pawprint Logo" class="find-header-logo">
            <h1 class="find-title">Find a missing pet</h1>
            <div class="find-sub-pill">upload a lost pet</div>
        </div>
    </header>
    <main class="find-main">
        <div id="find-notification-dialog" class="find-notification-dialog" style="display:none;">
            <span id="find-notification-dialog-text"></span>
        </div>
      <form class="find-form" id="findForm" action="see-matches.php" method="post" enctype="multipart/form-data">
          <input type="hidden" name="client_start_ms" id="client_start_ms" value="">
            <label for="pet-image" class="find-label">Upload Pet Image: <span style="color:red">*</span></label>
            <div class="find-image-upload">
                <input type="file" id="pet-image" name="pet-image" accept="image/*" class="find-input" style="display:none;">
                <label for="pet-image" class="find-image-upload-label" style="cursor:pointer;">
                    <span id="find-image-placeholder">Click to upload or drag an image here</span>
                    <img id="find-image-preview" src="" alt="Preview" style="display:none;max-width:100%;max-height:180px;border-radius:12px;box-shadow:0 2px 8px rgba(60,90,200,0.10);margin-top:8px;">
                </label>
            </div>

            <label for="pet-type" class="find-label">Pet Type:</label>
            <select id="pet-type" name="pet-type" class="find-input">
                <option value="">Auto-Detect</option>
                <option value="Dog">Dog</option>
                <option value="Cat">Cat</option>
            </select>
            <div class="find-hint">Filter results by pet type</div>

            <div class="find-preprocess-section">
                <label class="find-label" for="preprocess" style="margin-bottom:20px;"><b>Image Pre-processing</b></label>
                <div class="find-checkbox-row" style="margin-bottom:10px;">
                    <input type="checkbox" id="preprocess" name="preprocess" class="find-checkbox" checked>
                    <label for="preprocess" class="find-checkbox-label">Enable advanced image pre-processing</label>
                </div>
                <div class="find-preprocess-desc" style="margin-bottom:10px;">
                    Image pre-processing improves detection accuracy by normalizing images into a standard body/face crop of pet and resizing it to 224x224.<br>
                    <span style="display:block;margin-top:4px;">Disable only if you experience issues.</span>
                </div>
            </div>
            <button type="submit" class="find-submit-btn">Submit</button>
        </form>
        <script>
        // Image upload preview logic
        const imageInput = document.getElementById('pet-image');
        const imagePreview = document.getElementById('find-image-preview');
        const imagePlaceholder = document.getElementById('find-image-placeholder');
        const imageLabel = document.querySelector('.find-image-upload-label');
        const imageUploadBox = document.querySelector('.find-image-upload');
        const findForm = document.getElementById('findForm');
        // Dialog elements
        const findNotificationDialog = document.getElementById('find-notification-dialog');
        const findNotificationDialogText = document.getElementById('find-notification-dialog-text');
        const findLoadingOverlay = document.getElementById('find-loading-overlay');
        let dialogTimeout = null;

        function showDialog(message) {
            findNotificationDialogText.textContent = message;
            findNotificationDialog.classList.remove('hide');
            findNotificationDialog.classList.add('show');
            findNotificationDialog.style.display = 'flex';
            if (dialogTimeout) clearTimeout(dialogTimeout);
            dialogTimeout = setTimeout(hideDialog, 2500);
        }
        function hideDialog() {
            findNotificationDialog.classList.remove('show');
            findNotificationDialog.classList.add('hide');
            setTimeout(() => {
                findNotificationDialog.style.display = 'none';
                findNotificationDialogText.textContent = '';
            }, 350);
        }

        function hasImageUploaded() {
            return imageInput.files && imageInput.files.length > 0 && imageInput.files[0].type.startsWith('image/');
        }

        imageInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file && file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = function(ev) {
                    imagePreview.src = ev.target.result;
                    imagePreview.style.display = 'block';
                    imagePlaceholder.style.display = 'none';
                };
                reader.readAsDataURL(file);
                hideDialog();
            } else {
                imagePreview.src = '';
                imagePreview.style.display = 'none';
                imagePlaceholder.style.display = 'block';
            }
        });

        // Drag and drop support
        imageUploadBox.addEventListener('dragover', function(e) {
            e.preventDefault();
            imageUploadBox.style.borderColor = '#3867d6';
        });
        imageUploadBox.addEventListener('dragleave', function(e) {
            e.preventDefault();
            imageUploadBox.style.borderColor = '#b3c6ff';
        });
        imageUploadBox.addEventListener('drop', function(e) {
            e.preventDefault();
            imageUploadBox.style.borderColor = '#b3c6ff';
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type.startsWith('image/')) {
                imageInput.files = files;
                const event = new Event('change');
                imageInput.dispatchEvent(event);
            } else {
                imageInput.value = '';
                imagePreview.src = '';
                imagePreview.style.display = 'none';
                imagePlaceholder.style.display = 'block';
                showDialog('Please upload a valid image file.');
            }
        });

        // Dialog notification on submit if image missing
        findForm.addEventListener('submit', function(e) {
            if (!hasImageUploaded()) {
                e.preventDefault();
                showDialog('Please upload a pet image before submitting.');
            } else {
                hideDialog();
                // Show loading overlay with staged progress simulation
                findLoadingOverlay.style.display = 'flex';
                // Record client start timestamp (ms since epoch)
                const tsField = document.getElementById('client_start_ms');
                if(tsField){ tsField.value = Date.now(); }
                findForm.querySelector('button[type="submit"]').disabled = true;
                simulateFindProgress();
            }
        });

        // Staged description simulation (client-side only; real server steps happen after submit)
        const progressStage = document.getElementById('find-progress-stage');
        const progressDetail = document.getElementById('find-progress-detail');

        function simulateFindProgress(){
            const stages = [
                {t:0,    stage:'Initializing', detail:'Preparing preprocessing pipeline...'},
                {t:3000,  stage:'Reading Image', detail:'Loading file & validating format...'},
                {t:4000,  stage:'Detecting Pet', detail:'Running YOLO face/body detection...'},
                {t:5000, stage:'Cropping', detail:'Extracting detected region & cleaning background...'},
                {t:4000, stage:'Normalizing', detail:'Resizing to 224x224 and enhancing quality...'},
                {t:6000, stage:'Preparing Matches', detail:'Comparing against stored pet dataset...'},
                {t:3600, stage:'Finalizing', detail:'Sorting and formatting match results...'}
            ];
            stages.forEach(s=>{
                setTimeout(()=>{
                    progressStage.textContent = s.stage;
                    progressDetail.textContent = s.detail;
                }, s.t);
            });
            setTimeout(()=>{
                progressDetail.textContent = 'Almost done...';
            }, 5000);
        }
        </script>
    </main>
    <style>
    #find-loading-overlay {
        transition: opacity 0.3s;
        font-family: inherit;
    }
    .find-spinner svg {
        animation: find-spin 1s linear infinite;
    }
    @keyframes find-spin {
        100% { transform: rotate(360deg); }
    }
    </style>
</body>
</html>
