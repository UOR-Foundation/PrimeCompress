import React from 'react';
import { HashRouter, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Navbar from './components/common/Navbar';
import HomePage from './pages/HomePage';
import TestPage from './pages/TestPage';
import AboutPage from './pages/AboutPage';
import './App.css';

// Create a theme instance
const theme = createTheme({
  palette: {
    primary: {
      main: '#3f51b5',
    },
    secondary: {
      main: '#f50057',
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <HashRouter>
        <div className="App">
          <Navbar />
          <main style={{ padding: '20px' }}>
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/test" element={<TestPage />} />
              <Route path="/about" element={<AboutPage />} />
            </Routes>
          </main>
        </div>
      </HashRouter>
    </ThemeProvider>
  );
}

export default App;
