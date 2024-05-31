import React from 'react'
import Header from '../Components/Header'
import { Link } from 'react-router-dom'

function Choose() {
  return (
    <div>
        <Header></Header>
        <div className='container'>      
            <div className="grid grid-cols-2 grid-rows-1 gap-4 items-center justify-center mx-auto">
                <Link to="/plate" className='item-center mx-auto'>
                    <div className='w-2/3 p-2 border-2 bg-gray-50 mt-10 mx-auto w-64 h-72 shadow-lg'>
                        <h1 className='text-black font-montserrat m-2 text-center'>Plate Detection</h1>
                        <img src="https://www.protoclea.com/wp-content/uploads/2019/10/Automatic-number-plate-recognition-icon.png" alt="" />
                    </div>
                </Link>
                <Link to="/object" className='item-center mx-auto'>
                    <div className='w-2/3 p-2 border-2 bg-gray-50 mt-10 mx-auto w-64 h-72 shadow-lg'>
                        <h1 className='text-black font-montserrat m-2 text-center '>Object Detection</h1>
                        <img src="https://cdn4.iconfinder.com/data/icons/self-driving-car-outline/64/Lane_Detection-512.png" alt="" />
                    </div>
                </Link>
            </div>
        </div>
    </div>
  )
}

export default Choose
