import { BrowserRouter,Route,Routes } from "react-router-dom"
import Home from "./Pages/Home"
import Choose from "./Pages/Choose"
import ObjectDetection from "./Pages/ObjectDetection"
import PlateReader from "./Pages/PlateReader"
import Login from "./Pages/Login"
import Signup from "./Pages/SignUp"
import ProtectedRoute from './ProtectedRoot';


function App() {

  return (
    <>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home></Home>} />
          <Route
          path="/choose"
          element={
            <ProtectedRoute>
              <Choose />
            </ProtectedRoute>
          }
        />
          <Route path="*" element={<h1>404 Not Found</h1>}></Route>
          <Route path ="/plate" element={<PlateReader></PlateReader>}></Route>
          <Route path="/object" element={<ObjectDetection></ObjectDetection>}></Route>
          <Route path="/login" element={<Login></Login>}></Route>
          <Route path="/signup" element={<Signup></Signup>}></Route>
        </Routes>
      </BrowserRouter>
    </>
  )
}

export default App
