// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyDTs-OezJs4xY5Lw3fV4M0JFtmku6KqK18",
  authDomain: "vision-a0cc7.firebaseapp.com",
  projectId: "vision-a0cc7",
  storageBucket: "vision-a0cc7.appspot.com",
  messagingSenderId: "948226311463",
  appId: "1:948226311463:web:e70b8fcac8d25268f61502"
};
// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Firebase Authentication and get a reference to the service
export const auth = getAuth(app);
export default app;