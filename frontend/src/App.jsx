import { BrowserRouter,Route,Routes } from "react-router-dom"
import Home from "./Pages/Home"
import Choose from "./Pages/Choose"
import ObjectDetection from "./Pages/ObjectDetection"
import PlateReader from "./Pages/PlateReader"
function App() {

  return (
    <>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home></Home>} />
          <Route path="/choose" element={<Choose></Choose>}></Route>
          <Route path="*" element={<h1>404 Not Found</h1>}></Route>
          <Route path ="/plate" element={<PlateReader></PlateReader>}></Route>
          <Route path="/object" element={<ObjectDetection></ObjectDetection>}></Route>
        </Routes>
      </BrowserRouter>
    </>
  )
}

export default App
