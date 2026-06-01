async function uploadImage(formData: FormData) {
    const response = await fetch("http://127.0.0.1:8000/prediction", {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Upload failed: ${response.status} ${errorText}`);
    }

    return response.json();
}

export default uploadImage;