import React from 'react';
import { Typography, Box, Container, Grid, Paper, Button, Card, CardContent } from '@mui/material';
import { styled } from '@mui/material/styles';
import { Link as RouterLink } from 'react-router-dom';

const HeroSection = styled(Box)(({ theme }) => ({
  background: `linear-gradient(45deg, ${theme.palette.primary.dark} 30%, ${theme.palette.primary.main} 90%)`,
  color: theme.palette.primary.contrastText,
  padding: theme.spacing(8, 0),
  marginBottom: theme.spacing(4),
}));

const FeatureCard = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  transition: 'transform 0.2s',
  '&:hover': {
    transform: 'translateY(-5px)',
    boxShadow: theme.shadows[6],
  },
}));

const CodeBlock = styled(Box)(({ theme }) => ({
  backgroundColor: theme.palette.grey[900],
  color: theme.palette.common.white,
  padding: theme.spacing(2),
  borderRadius: theme.shape.borderRadius,
  overflowX: 'auto',
  fontFamily: 'monospace',
  marginTop: theme.spacing(2),
  marginBottom: theme.spacing(2),
}));

const HomePage: React.FC = () => {
  return (
    <>
      <HeroSection>
        <Container>
          <Typography variant="h2" component="h1" gutterBottom>
            PrimeCompress
          </Typography>
          <Typography variant="h5" component="h2" gutterBottom>
            Advanced compression library with multiple strategies for optimal compression across diverse data types
          </Typography>
          <Box mt={4}>
            <Button 
              variant="contained" 
              color="secondary" 
              size="large"
              component={RouterLink}
              to="/test"
              sx={{ mr: 2 }}
            >
              Test Compression
            </Button>
            <Button 
              variant="outlined" 
              color="inherit"
              size="large"
              href="https://github.com/UOR-Foundation/PrimeCompress"
              target="_blank"
              rel="noopener noreferrer"
            >
              View on GitHub
            </Button>
          </Box>
        </Container>
      </HeroSection>

      <Container>
        <Box mb={6}>
          <Typography variant="h4" component="h2" gutterBottom>
            Multiple Compression Strategies
          </Typography>
          <Typography variant="body1" paragraph>
            PrimeCompress utilizes advanced algorithms to choose the optimal compression strategy based on your data characteristics.
          </Typography>
          
          <Grid container spacing={3} mt={2}>
            <Grid item xs={12} sm={6} md={4}>
              <FeatureCard>
                <Typography variant="h6" gutterBottom>Pattern Recognition</Typography>
                <Typography variant="body2">
                  Identifies repeating patterns in binary data for maximum compression ratio.
                </Typography>
              </FeatureCard>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <FeatureCard>
                <Typography variant="h6" gutterBottom>Sequential Compression</Typography>
                <Typography variant="body2">
                  Optimized for arithmetic sequences and i%N patterns common in generated data.
                </Typography>
              </FeatureCard>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <FeatureCard>
                <Typography variant="h6" gutterBottom>Spectral Compression</Typography>
                <Typography variant="body2">
                  Specialized for sinusoidal and wave-like data patterns using spectral analysis.
                </Typography>
              </FeatureCard>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <FeatureCard>
                <Typography variant="h6" gutterBottom>Dictionary Compression</Typography>
                <Typography variant="body2">
                  Enhanced frequency-optimized dictionary compression with Huffman coding for text data.
                </Typography>
              </FeatureCard>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <FeatureCard>
                <Typography variant="h6" gutterBottom>Block-Based Compression</Typography>
                <Typography variant="body2">
                  Splits large datasets into optimal blocks, applying the best strategy for each section.
                </Typography>
              </FeatureCard>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <FeatureCard>
                <Typography variant="h6" gutterBottom>High-Entropy Optimization</Typography>
                <Typography variant="body2">
                  Fast-path processing for incompressible random data to minimize overhead.
                </Typography>
              </FeatureCard>
            </Grid>
          </Grid>
        </Box>

        <Box mb={6}>
          <Typography variant="h4" component="h2" gutterBottom>
            Easy Integration
          </Typography>
          <Typography variant="body1" paragraph>
            Integrate PrimeCompress into your projects with just a few lines of code:
          </Typography>
          
          <CodeBlock>
            <pre>{`// When installed from npm
const compression = require('@uor-foundation/prime-compress');

// Compress data
const compressedData = compression.compress(data);

// Decompress data
const originalData = compression.decompress(compressedData);

// Compress with a specific strategy
const compressedWithStrategy = compression.compressWithStrategy(data, 'dictionary');`}</pre>
          </CodeBlock>
          
          <Button 
            variant="contained" 
            color="primary"
            component={RouterLink}
            to="/about"
            sx={{ mt: 2 }}
          >
            Learn More
          </Button>
        </Box>

        <Box mb={6}>
          <Typography variant="h4" component="h2" gutterBottom>
            Performance Comparison
          </Typography>
          <Typography variant="body1" paragraph>
            PrimeCompress outperforms standard compression techniques for specialized data types:
          </Typography>
          
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Benchmark Results
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Improvement over standard compression techniques:
              </Typography>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2">Text Data</Typography>
                <Typography variant="body2" color="success.main">+185.83%</Typography>
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2">Mixed Data (Block-Based)</Typography>
                <Typography variant="body2" color="success.main">+210.59%</Typography>
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2">Sine Wave</Typography>
                <Typography variant="body2" color="success.main">+60.00%</Typography>
              </Box>
            </CardContent>
          </Card>
          
          <Button 
            variant="outlined" 
            color="primary"
            component={RouterLink}
            to="/test"
          >
            Run Test Comparison
          </Button>
        </Box>
      </Container>
    </>
  );
};

export default HomePage;