import { useState } from 'react';
import {
  Box,
  Button,
  Container,
  LinearProgress,
  TextField,
  Typography
} from '@mui/material';

export default function Home() {
  const [file, setFile] = useState(null);
  const [summary, setSummary] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a PDF file.');
      return;
    }
    setError('');
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        const errorData = await res.json();
        setError(errorData.detail || 'An error occurred.');
        setLoading(false);
        return;
      }
      const data = await res.json();
      setSummary(data.summary);
    } catch (err) {
      setError('An error occurred.');
    }
    setLoading(false);
  };

  return (
    <Container maxWidth="md">
      <Typography variant="h4" align="center" gutterBottom>
        PDF Summarizer
      </Typography>
      {error && <Typography color="error">{error}</Typography>}
      <form onSubmit={handleSubmit}>
        <Box display="flex" alignItems="center" my={2}>
          <Button variant="contained" component="label">
            Select PDF
            <input type="file" hidden accept="application/pdf" onChange={handleFileChange} />
          </Button>
          <Typography variant="body1" ml={2}>
            {file ? file.name : 'No file selected'}
          </Typography>
        </Box>
        <Button type="submit" variant="contained" color="primary" disabled={loading}>
          Upload and Summarize
        </Button>
      </form>
      {loading && <LinearProgress />}
      {summary && (
        <Box mt={4}>
          <Typography variant="h5">Summary</Typography>
          <Typography variant="body1">{summary}</Typography>
        </Box>
      )}
    </Container>
  );
}