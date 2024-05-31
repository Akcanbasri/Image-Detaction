import React from 'react'

function Desc() {
  return (
    <div className='container pt-4 pb-4'>
      <div className='font-montserrat'>
        <h1 className='text-3xl text-center mt-4'>Why use Vision Detect</h1>
        <div className='flex justify-center space-x-10 mt-6 pb-4'>
            <div className='mt-6'>
                <img src="https://cdn-icons-png.freepik.com/512/8596/8596602.png" className='w-16 mx-auto' alt="" />
                <h1 className='text-center font-bold'>High Detection Accuracy</h1>
                <p className='text-center text-xs w-64'>High detection accuracy means a system or model correctly identifies most relevant inputs with minimal errors.</p>
            </div>
            <div className='mt-6'>
                <img src="https://cdn-icons-png.freepik.com/512/8948/8948529.png" className='w-16 mx-auto' alt="" />
                <h1 className='text-center font-bold'>High Detection Accuracy</h1>
                <p className='text-center text-xs w-64'>Confidentiality guaranteed means that information will be kept private and secure, with access restricted to authorized individuals only.</p>
            </div>
            <div className='mt-6'>
                <img src="https://cdn-icons-png.flaticon.com/512/7690/7690595.png" className='w-16 mx-auto' alt="" />
                <h1 className='text-center font-bold'>Advanced Algorithms</h1>
                <p className='text-center text-xs w-64'>Advanced algorithms are complex computational methods designed to solve problems efficiently and effectively.</p>
            </div>
        </div>
      </div>
    </div>
  )
}

export default Desc
