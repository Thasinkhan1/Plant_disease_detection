function predict() {
    const fileInput = document.getElementById("imageInput");
    const file = fileInput.files[0];
  
    if (!file) return alert("Please choose an image!");
  
    const formData = new FormData();
    formData.append("file", file);
  
    document.getElementById("preview").classList.remove("hidden");
    document.getElementById("imagePreview").src = URL.createObjectURL(file);
  
    fetch("http://localhost:8000/predict", {
      method: "POST",
      body: formData
    })
      .then(res => res.json())
      .then(data => {
        document.getElementById("result").classList.remove("hidden");
        document.getElementById("diseaseResult").textContent = `Disease: ${data.prediction} (${data.confidence}%)`;
        document.getElementById("adviceText").textContent = `Advice: ${data.advice}`;
      })
      .catch(err => {
        alert("Prediction failed.");
        console.error(err);
      });
  }
  