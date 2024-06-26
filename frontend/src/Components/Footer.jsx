import React from 'react'

function Footer() {
  return (
    <div className='bg-gray-100 pt-10 pb-10'>
      <div className='flex container'>
        <div className='mt-4'>
          <h1 className='font-montserrat 3text-xl'>Vision Detect</h1>
          <p className='text-xs'>Vision Detect is a platform that uses advanced algorithms to detect and recognize objects in images.</p>
        </div>
        <div className='ml-20 mt-4'>
          <h1 className='font-montserrat'>Quick Links</h1>
          <div className='flex-col'>
            <h1 className='text-xs'>About</h1>
            <h1 className='text-xs'>Contact</h1>
            <h1 className='text-xs'>Price</h1>
          </div>
        </div>
        <div className='ml-20 mt-4'>
          <h1 className='font-montserrat'>Contact Us</h1>
          <p className='text-xs'>visiondetect@gmail.com</p>
      </div>
    </div>
    </div>
  )
}

export default Footer
