function previewImage(input, previewId) {
    const preview = document.getElementById(previewId);
    const file = input.files[0];
    
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.innerHTML = `<img src="${e.target.result}" class="max-h-[200px] mx-auto">`;
        }
        reader.readAsDataURL(file);
    }
}

function compareImages() {
    const image1 = document.getElementById('image1').files[0];
    const image2 = document.getElementById('image2').files[0];

    if (!image1 || !image2) {
        alert('Please upload both images first! 😊');
        return;
    }

    const formData = new FormData();
    formData.append('image1', image1);
    formData.append('image2', image2);

    fetch('http://127.0.0.1:8000/compare_images/', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const similarity = data.similarity_score;

        // Show result section
        document.getElementById('result').classList.remove('hidden');
        document.getElementById('similarity-score').innerText = `${similarity.toFixed(2)}%`;

        // Set result message based on similarity
        const resultMessage = document.getElementById('result-message');
        if (similarity >= 90) {
            resultMessage.innerText = "They're almost the same! 🎉";
            resultMessage.className = 'text-green-500 text-xl font-fredoka text-center mt-4';
        } else if (similarity >= 70) {
            resultMessage.innerText = "They're quite similar! 😊";
            resultMessage.className = 'text-blue-500 text-xl font-fredoka text-center mt-4';
        } else if (similarity >= 40) {
            resultMessage.innerText = "They're a bit different! 🤔";
            resultMessage.className = 'text-yellow-500 text-xl font-fredoka text-center mt-4';
        } else {
            resultMessage.innerText = "They're very different! 😅";
            resultMessage.className = 'text-red-500 text-xl font-fredoka text-center mt-4';
        }
        const gauge = document.getElementById('similarity-gauge');
        gauge.style.background = `conic-gradient(#8B5CF6 ${similarity * 3.6}deg, #E5E7EB ${similarity * 3.6}deg)`;
        gauge.style.borderRadius = '50%';
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Something went wrong! Please try again.');
    });
}