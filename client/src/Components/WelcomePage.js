import React from "react";
import { useNavigate } from "react-router-dom";

const WelcomePage = () => {
  const navigate = useNavigate();

  return (
    <div className="fullscreen-image">
      <button className="login-top-right" onClick={() => navigate('/login')}>
        Login
      </button>
      {/* Add any other overlay content here */}
    </div>
  );
};

export default WelcomePage;
