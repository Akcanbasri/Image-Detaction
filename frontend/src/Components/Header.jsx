import React from 'react'
import logo2 from "../assets/logo2.webp"
import { FaRegUserCircle } from "react-icons/fa";
import { Link } from 'react-router-dom';
import { onAuthStateChanged,getAuth,signOut  } from "firebase/auth";
import { auth } from '../firebase';
import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

function Header() {

  const [isAuthenticated, setIsAuthenticated] = useState(false);

  useEffect(() => {
    const auth = getAuth();
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      if (user) {
        setIsAuthenticated(true);
      } else {
        setIsAuthenticated(false);
      }
    });

    return () => unsubscribe();
  }, []);

  const navigate = useNavigate();
 
    const handleLogout = () => {               
        signOut(auth).then(() => {
        // Sign-out successful.
            navigate("/");
            console.log("Signed out successfully")
        }).catch((error) => {
        // An error happened.
        });
    }

  return (
    <div className='bg-[#02182F]'>
      <div className='container font-montserrat justify-between flex py-2 text-white'>
        <img src={logo2} className='w-20 h-20 ' alt="" />
         <div className='flex mt-6'>
            <Link to="/" className='px-3'>Home</Link>
            <Link href="#" className='px-3'>About</Link>
            <Link href="#" className='px-3'>Services</Link>
            <Link href="#" className='px-3'>Contact</Link>
         </div>
         {
            isAuthenticated ? (
              <button onClick={handleLogout}>logout</button>
            ) : (
              <div className='flex mt-4'>
              <div className='flex  border-2 h-10 w-24 flex items-center justify-between cursor-pointer rounded hover:bg-white duration-200 hover:text-black'>
                  <FaRegUserCircle className='absolute mt-0.5 ml-2 h-10 w-6' />
                  <Link to="/login" className='ml-10'>Login</Link>
              </div>
              
              <div className='border ml-4 mt-1 h-8'>
    
              </div>
              <Link to="/signup" className='mt-2 ml-4'>Register</Link>
             </div>
            )
          }

      </div>
    </div>
  )
}

export default Header
