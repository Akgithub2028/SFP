import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import Submit from "./pages/Submit";
import About from "./pages/About";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/submit" element={<Submit />} />
        <Route path="/about" element={<About />} />
      </Routes>
    </BrowserRouter>
  );
}
