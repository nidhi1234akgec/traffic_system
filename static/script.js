document.getElementById('prediction-form').addEventListener('submit', function(event) {
    event.preventDefault();

    const location = document.getElementById('location').value;
    const time = document.getElementById('time').value;
    const weather = document.getElementById('weather').value;

    // Create the payload
    const payload = {
        location: location,
        time: time,
        weather: weather
    };

    // Send the data to the backend using fetch
    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result-container').classList.remove('hidden');
        document.getElementById('prediction-result').innerText = `Predicted Traffic Volume: ${data.prediction}`;
    })
    .catch(error => {
        console.error('Error:', error);
    });
});