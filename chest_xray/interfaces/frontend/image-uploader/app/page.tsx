"use client"
import { SubmitEvent } from 'react';
import uploadImage from '../services/uploadImage';



export default function StartPage() {

  const handleSubmit = async (event: SubmitEvent<HTMLFormElement>) => {
    event.preventDefault();
    console.log("Form submitted");
    const formData = new FormData(event.currentTarget);

    try {
      const result = await uploadImage(formData);
      console.log("Upload result:", result);
    } catch (error) {
      console.error("Upload failed:", error);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <div className="upload-wrapper flex flex-col items-center gap-4">
        <h1 className="text-2xl font-bold">Upload your chest X-ray image</h1>
        <form className='flex flex-col' onSubmit={handleSubmit}>
          <input className="file-input" type="file" accept="image/*" name="file" />
          <button className="submit-button mt-4" type="submit">Upload</button>
        </form>
      </div>
    </main>
  );
}
