import React from 'react'

function Detail() {
  return (
    <div className='bg-[#02182F] min-w-full mt-32 p-4 pb-10'>
      <div className='container text-white font-montserrat'>
      <h1 className='text-3xl p-2'>Object Detection</h1>
        <div className='flex mt-8'>   
            <p className='w-2/3 mr-6 '>Object detection is a computer vision technique that involves identifying and locating objects within an image or video. It combines image classification and object localization, enabling systems to recognize and determine the positions of multiple objects simultaneously. Techniques like convolutional neural networks (CNNs), region-based CNNs (R-CNN), and YOLO (You Only Look Once) are commonly used for object detection. Applications include autonomous driving, surveillance, and image annotation. </p>
            <img className='w-1/3' src="https://miro.medium.com/v2/resize:fit:896/1*4atR_wq8Wk4J4Z7JejfkCQ.png" alt="" />
        </div>
        <h1 className='text-3xl p-2 text-end mt-10'>Car Licanse Plate Detection</h1>
        <div className='flex mt-8'> 
            <img className='w-1/3' src="https://deepxhub.com/wp-content/uploads/2021/06/image-11.png" alt="" />
            <p className='w-2/3 ml-6 mt-4'>Car license plate detection is a specialized form of object detection that focuses on identifying and reading vehicle license plates from images or video. It typically involves steps like plate localization, character segmentation, and optical character recognition (OCR). This technology is widely used in applications such as traffic monitoring, toll collection, and automated parking systems. </p>
        </div>
      </div>
    </div>
  )
}

export default Detail
