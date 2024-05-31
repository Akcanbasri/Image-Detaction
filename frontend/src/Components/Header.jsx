import React from 'react'
import logo2 from "../assets/logo2.webp"
import { FaRegUserCircle } from "react-icons/fa";

function Header() {
  return (
    <div className='bg-[#02182F]'>
      <div className='container font-montserrat justify-between flex py-2 text-white'>
        <img src={logo2} className='w-20 h-20 ' alt="" />
         <div className='flex mt-6'>
            <a href="#" className='px-3'>Home</a>
            <a href="#" className='px-3'>About</a>
            <a href="#" className='px-3'>Services</a>
            <a href="#" className='px-3'>Contact</a>
         </div>
         <div className='flex mt-4'>
          <div className='flex  border-2 h-10 w-24 flex items-center justify-between cursor-pointer rounded hover:bg-white duration-200 hover:text-black'>
              <FaRegUserCircle className='absolute mt-0.5 ml-2 h-10 w-6' />
              <a href="#" className='ml-10'>Login</a>
          </div>
          <div className='border ml-4 mt-1 h-8'>

          </div>
          <h1 className='mt-2 ml-4'>Register</h1>
         </div>
      </div>
    </div>
  )
}

export default Header
