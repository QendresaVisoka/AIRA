import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const LoginPage = () => {
  // State variables for username, password, and error message
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  // Function to handle login
  const handleLogin = (e) => {
    e.preventDefault();

    // Check if the username and password are correct
    // In a real application, you would send these to a server for validation
    // Here logic is hardcoded for demonstration purposes
    if (username === 'admin' && password === 'AIRA') {
      setError('');
      // Redirect to the upload page
      navigate('/upload'); 
    } else {
      setError('Invalid credentials. Try "admin" / "AIRA"');
    }
  };

  return (
    <div
      style={{
        backgroundImage: 'url("/login-background.png")',
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        backgroundRepeat: 'no-repeat',
        width: '100vw',
        height: '100vh',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        position: 'relative',
      }}
    >

      <div className="login-box">

        <div>
          <button className='top-left-button' onClick={() => navigate('/')}>
            Back
          </button>
        </div>

        <h2>Please login to continue</h2>
        <form onSubmit={handleLogin}>
          <input
            type="text"
            placeholder="Username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
            style={{ width: '100%', marginBottom: '1rem', padding: '10px' }}
          />

          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            style={{ width: '100%', marginBottom: '1rem', padding: '10px' }}
          />

          {error && <p style={{ color: 'red', marginBottom: '1rem' }}>{error}</p>}

          <button
            type="submit"
            disabled={!username || !password}
            style={{ width: '105%', padding: '10px' }}
          >
            Login
          </button>
        </form>

      </div>
    </div>
  );
};

export default LoginPage;
