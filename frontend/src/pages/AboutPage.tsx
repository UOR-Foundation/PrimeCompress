import React from 'react';
import { Typography, Box, Container, Grid, Paper, Divider, List, ListItem, ListItemText, Link } from '@mui/material';
import { styled } from '@mui/material/styles';

const SectionPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  marginBottom: theme.spacing(3),
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

const AboutPage: React.FC = () => {
  return (
    <Container>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          About PrimeCompress
        </Typography>
        <Typography variant="body1">
          PrimeCompress is an advanced compression library featuring multiple strategies
          for optimal compression across diverse data types.
        </Typography>
      </Box>

      <SectionPaper>
        <Typography variant="h5" gutterBottom>Features</Typography>
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <List>
              <ListItem>
                <ListItemText 
                  primary="Strategy Selection" 
                  secondary="Intelligent selection of compression strategy based on data characteristics"
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="Block-Based Compression" 
                  secondary="Splits larger data into blocks, applying optimal compression for each block"
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="True Huffman Encoding" 
                  secondary="Dictionary compression with Huffman coding for text data"
                />
              </ListItem>
            </List>
          </Grid>
          <Grid item xs={12} md={6}>
            <List>
              <ListItem>
                <ListItemText 
                  primary="Fast Path for High-Entropy Data" 
                  secondary="Optimized handling of random or incompressible data"
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="Pattern Detection" 
                  secondary="Recognition of constants, patterns, sequences, and spectral characteristics"
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="Perfect Reconstruction" 
                  secondary="Guaranteed exact data reconstruction with checksum validation"
                />
              </ListItem>
            </List>
          </Grid>
        </Grid>
      </SectionPaper>

      <SectionPaper>
        <Typography variant="h5" gutterBottom>Installation</Typography>
        <Typography variant="body1" paragraph>
          Install PrimeCompress from GitHub Packages:
        </Typography>
        <CodeBlock>
          <pre>{`# Configure npm to use GitHub Packages
echo "@uor-foundation:registry=https://npm.pkg.github.com" >> .npmrc

# Install the package
npm install @uor-foundation/prime-compress`}</pre>
        </CodeBlock>
        <Typography variant="body1" paragraph>
          Or install from source:
        </Typography>
        <CodeBlock>
          <pre>{`git clone https://github.com/UOR-Foundation/PrimeCompress.git
cd PrimeCompress
npm install`}</pre>
        </CodeBlock>
      </SectionPaper>

      <SectionPaper>
        <Typography variant="h5" gutterBottom>Usage</Typography>
        <Typography variant="body1" paragraph>
          Basic usage of the library:
        </Typography>
        <CodeBlock>
          <pre>{`const compression = require('@uor-foundation/prime-compress');

// Compress data
const compressedData = compression.compress(data);

// Decompress data
const originalData = compression.decompress(compressedData);

// Compress with a specific strategy
const compressedWithStrategy = compression.compressWithStrategy(data, 'dictionary');

// Analyze data to determine optimal compression strategy
const analysis = compression.analyzeCompression(data);
console.log(\`Recommended strategy: \${analysis.recommendedStrategy}\`);`}</pre>
        </CodeBlock>
      </SectionPaper>

      <SectionPaper>
        <Typography variant="h5" gutterBottom>Compression Strategies</Typography>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Paper variant="outlined" sx={{ p: 2, height: '100%' }}>
              <Typography variant="h6" gutterBottom>Zeros Compression</Typography>
              <Divider sx={{ mb: 1 }} />
              <Typography variant="body2">
                Optimized for constant data (all bytes are the same). Achieves the maximum possible
                compression ratio by storing only the constant value and the original data length.
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} md={6}>
            <Paper variant="outlined" sx={{ p: 2, height: '100%' }}>
              <Typography variant="h6" gutterBottom>Pattern Compression</Typography>
              <Divider sx={{ mb: 1 }} />
              <Typography variant="body2">
                Identifies and compresses repeating patterns in binary data. Particularly effective
                for data with recurring byte sequences, offering high compression ratios.
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} md={6}>
            <Paper variant="outlined" sx={{ p: 2, height: '100%' }}>
              <Typography variant="h6" gutterBottom>Sequential Compression</Typography>
              <Divider sx={{ mb: 1 }} />
              <Typography variant="body2">
                Specialized for arithmetic sequences and i%N patterns. Stores only the formula and parameters
                instead of the full data, achieving excellent compression for ordered sequences.
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} md={6}>
            <Paper variant="outlined" sx={{ p: 2, height: '100%' }}>
              <Typography variant="h6" gutterBottom>Spectral Compression</Typography>
              <Divider sx={{ mb: 1 }} />
              <Typography variant="body2">
                Optimized for sinusoidal and wave-like data using spectral analysis. Stores the dominant
                frequency components, achieving high compression for audio-like and oscillating data.
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} md={6}>
            <Paper variant="outlined" sx={{ p: 2, height: '100%' }}>
              <Typography variant="h6" gutterBottom>Dictionary Compression</Typography>
              <Divider sx={{ mb: 1 }} />
              <Typography variant="body2">
                Enhanced frequency-optimized dictionary compression with Huffman coding.
                Ideal for text data, storing frequent tokens efficiently to maximize compression.
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} md={6}>
            <Paper variant="outlined" sx={{ p: 2, height: '100%' }}>
              <Typography variant="h6" gutterBottom>Statistical Compression</Typography>
              <Divider sx={{ mb: 1 }} />
              <Typography variant="body2">
                Fallback method for high-entropy data, using statistical models to optimize compression
                when other strategies don't yield significant improvements.
              </Typography>
            </Paper>
          </Grid>
        </Grid>
      </SectionPaper>

      <SectionPaper>
        <Typography variant="h5" gutterBottom>Advanced Features</Typography>
        <Typography variant="h6" sx={{ mt: 2 }}>Block-Based Compression</Typography>
        <Typography variant="body1" paragraph>
          For larger datasets, PrimeCompress automatically applies block-based compression,
          dividing data into optimal blocks and compressing each with the best strategy.
        </Typography>
        <CodeBlock>
          <pre>{`// Enable block-based compression (enabled by default for data > 4KB)
const compressedBlocks = compression.compress(largeData, { useBlocks: true });`}</pre>
        </CodeBlock>
        
        <Typography variant="h6" sx={{ mt: 2 }}>Enhanced Dictionary Compression</Typography>
        <Typography variant="body1" paragraph>
          Text data benefits from frequency-optimized dictionary compression with Huffman coding
          for further space savings.
        </Typography>
        <CodeBlock>
          <pre>{`// Force dictionary strategy with Huffman coding
const compressedText = compression.compressWithStrategy(textData, 'enhanced-dictionary');`}</pre>
        </CodeBlock>
      </SectionPaper>

      <SectionPaper>
        <Typography variant="h5" gutterBottom>Resources</Typography>
        <List>
          <ListItem>
            <ListItemText 
              primary={<Link href="https://github.com/UOR-Foundation/PrimeCompress" target="_blank" rel="noopener noreferrer">GitHub Repository</Link>}
              secondary="Source code and documentation"
            />
          </ListItem>
          <ListItem>
            <ListItemText 
              primary={<Link href="https://github.com/UOR-Foundation/PrimeCompress/issues" target="_blank" rel="noopener noreferrer">Issue Tracker</Link>}
              secondary="Report bugs and request features"
            />
          </ListItem>
          <ListItem>
            <ListItemText 
              primary="License"
              secondary="PrimeCompress is released under the MIT License"
            />
          </ListItem>
        </List>
      </SectionPaper>
    </Container>
  );
};

export default AboutPage;