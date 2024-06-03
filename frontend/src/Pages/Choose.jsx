import React from "react";
import Header from "../Components/Header";
import Footer from "../Components/Footer"; // Assuming you have a Footer component
import Lottie from "lottie-react";
import animationData from "../assets/car.json";
import animationnewData from "../assets/newcar.json";
import { Link } from "react-router-dom";

function Choose() {
  return (
    <div>
      <Header />
      <div className="bg-gray-100 min-h-screen flex flex-col items-center">
        <div className="container py-20">
          <h1 className="text-4xl font-bold mb-10 text-center">Choose an Option</h1>
          <div className="flex flex-col lg:flex-row items-center justify-center lg:space-x-10 space-y-10 lg:space-y-0">
            <Link to="/plate" className="group">
              <div className="flex flex-col items-center bg-white shadow-lg rounded-lg p-8 hover:bg-gray-200 transition cursor-pointer">
                <Lottie animationData={animationData} className="w-32 h-32 mb-4" />
                <h2 className="text-2xl font-semibold group-hover:text-blue-500">Plate Detection</h2>
                <p className="mt-2 text-center text-gray-600">
                  Detect and recognize license plates using advanced AI technology.
                </p>
              </div>
            </Link>
            <Link to="/object" className="group">
              <div className="flex flex-col items-center bg-white shadow-lg rounded-lg p-8 hover:bg-gray-200 transition cursor-pointer">
                <Lottie animationData={animationnewData} className="w-32 h-32 mb-4" />
                <h2 className="text-2xl font-semibold group-hover:text-blue-500">Object Detection</h2>
                <p className="mt-2 text-center text-gray-600">
                  Identify and categorize various objects with our powerful AI solutions.
                </p>
              </div>
            </Link>
          </div>
        </div>
      </div>
      <Footer /> {/* Assuming you have a Footer component */}
    </div>
  );
}

export default Choose;
