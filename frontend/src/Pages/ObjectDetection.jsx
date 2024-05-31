import React, { useState } from "react";
import axios from "axios";
import Header from "../Components/Header";
import Footer from "../Components/Footer";

function PlateReader() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [resultImage, setResultImage] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!selectedFile) {
      setError("Please select a file");
      return;
    }

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const response = await axios.post(
        "http://localhost:5000/detect_car",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
          responseType: "blob", // To handle binary data (image)
        }
      );

      const imageBlob = response.data;
      const imageObjectURL = URL.createObjectURL(imageBlob);
      setResultImage(imageObjectURL);
      setError(null);
    } catch (err) {
      setError("An error occurred: " + err.response.data.error);
      setResultImage(null);
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <div className="flex-grow container mx-auto p-4">
        <div className="flex flex-col md:flex-row items-center justify-center space-y-4 md:space-y-0 md:space-x-4">
          <form
            onSubmit={handleSubmit}
            className="bg-white p-6 rounded-lg shadow-md w-full max-w-md"
          >
            <h1 className="text-2xl font-bold text-center mb-4">
              Upload an Image to Read License Plate
            </h1>
            <input
              type="file"
              onChange={handleFileChange}
              className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
            />
            <button
              className="mt-4 w-full bg-blue-700 text-white py-2 rounded-xl shadow-lg hover:bg-blue-800 transition duration-200"
              type="submit"
            >
              Upload
            </button>
          </form>

          {error && <p className="text-red-500 text-center">{error}</p>}

          {resultImage && (
            <div className="w-full max-w-md mt-6 md:mt-0">
              <div className="border-2 rounded-lg p-4">
                <h2 className="text-center text-xl font-semibold mb-2">
                  Result:
                </h2>
                <img
                  src={resultImage}
                  className="mx-auto w-full h-auto rounded-lg"
                  alt="Result"
                />
              </div>
            </div>
          )}
        </div>
      </div>

      <Footer />
    </div>
  );
}

export default PlateReader;
