document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("uploadForm");
  const dropZone = document.getElementById("dropZone");
  const imageInput = document.getElementById("imageInput");
  const chooseButton = document.getElementById("chooseImageButton");
  const previewImage = document.getElementById("previewImage");
  const previewEmpty = document.getElementById("previewEmpty");
  const previewMeta = document.getElementById("previewMeta");
  const fileName = document.getElementById("fileName");
  const analyzeButton = document.getElementById("analyzeButton");

  if (!form || !dropZone || !imageInput) return;

  const maxFileSize = 10 * 1024 * 1024;
  const allowedTypes = ["image/jpeg", "image/png"];
  let previewUrl = null;

  function useFile(file) {
    if (!file) return;

    if (!allowedTypes.includes(file.type)) {
      alert("Please choose a JPG or PNG image.");
      imageInput.value = "";
      return;
    }

    if (file.size > maxFileSize) {
      alert("Please choose an image smaller than 10 MB.");
      imageInput.value = "";
      return;
    }

    // Keep the dropped file attached to the form for Flask upload.
    const transfer = new DataTransfer();
    transfer.items.add(file);
    imageInput.files = transfer.files;

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);

    previewImage.src = previewUrl;
    previewImage.hidden = false;
    previewEmpty.hidden = true;
    previewMeta.hidden = false;
    fileName.textContent = file.name;
    analyzeButton.disabled = false;
  }

  chooseButton.addEventListener("click", () => imageInput.click());

  dropZone.addEventListener("click", (event) => {
    if (event.target.closest("button")) return;
    imageInput.click();
  });

  dropZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      imageInput.click();
    }
  });

  imageInput.addEventListener("change", () => {
    useFile(imageInput.files[0]);
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.add("is-dragging");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.remove("is-dragging");
    });
  });

  dropZone.addEventListener("drop", (event) => {
    useFile(event.dataTransfer.files[0]);
  });

  form.addEventListener("submit", (event) => {
    if (!imageInput.files.length) {
      event.preventDefault();
      alert("Choose an image before starting the screening.");
      return;
    }

    analyzeButton.disabled = true;
    analyzeButton.textContent = "Analyzing image…";
  });
});
