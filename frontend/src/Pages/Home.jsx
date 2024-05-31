import React from "react";
import Header from "../Components/Header";
import Lottie from "lottie-react";
import animationData from "../assets/anim.json";
import Detail from "../Components/Detail";
import Desc from "../Components/Desc";
import Footer from "../Components/Footer";
import PlateReader from "./PlateReader";
import { Link } from "react-router-dom";

function Home() {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />   
        <div className="bg-white text-black py-20">
        <div className="container mx-auto px-6 flex flex-col lg:flex-row items-center justify-between">
          <div className="lg:w-1/2">
            <h1 className="text-4xl lg:text-6xl font-bold leading-tight mb-4">
              Welcome to Vision Detect
            </h1>
            <p className="text-md md:text-xl mb-6">
              Leverage cutting-edge AI technologies like Plate Detection and Object Detection to streamline your processes.
            </p>
            <Link to="/choose" className="bg-blue-500 text-white py-2 px-6 rounded-lg text-lg hover:bg-blue-600 transition">
              Get Started
            </Link>
          </div>
          <div className="lg:w-1/2 mt-10 lg:mt-0 ml-10">
            <Lottie animationData={animationData} />
          </div>
        </div>
      </div>
      <Footer />
    </div>
  );
}

export default Home;
