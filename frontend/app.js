const express = require('express');
const path = require('path');
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

const app = express();
const port = 3010;

// Set up the view engine
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// Set up static folders
app.use(express.static(path.join(__dirname, 'public')));

// Middleware to parse POST data
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

// Home Route (Landing page)
app.get('/', (req, res) => {
    res.render('index'); 
});

// Fetch file from Dagshub and send to Flask backend
app.post('/upload', async (req, res) => {
    console.log("Start calling Model in Upload");
    const dagshubFileUrl = req.body.fileUrl; // Assume the user provides the Dagshub file URL

    if (!dagshubFileUrl) {
        return res.render('index', { message: 'No file URL provided' });
    }

    try {
        // Fetch the file from Dagshub
        const fileResponse = await axios.get(dagshubFileUrl, { responseType: 'stream' });

        // Prepare form data to send to Flask backend
        const form = new FormData();
        form.append('file', fileResponse.data, path.basename(dagshubFileUrl));

        // Send the file to Flask backend
        const flaskResponse = await axios.post('http://fraud_detection_app-backend-1:5010/upload', form, {
            headers: form.getHeaders()
        });

        const predictions = flaskResponse.data.predictions;
        console.log(predictions);
        res.render('result', { predictions });
    } catch (error) {
        console.error('Error fetching file from Dagshub or sending to Flask:', error);
        res.render('index', { message: 'Error processing file' });
    }
});

// Fetch training file from Dagshub and send to Flask backend
app.post('/upload_train_file', async (req, res) => {
    console.log("Start calling Model in Retrain Upload Model");
    const dagshubFileUrl = req.body.trainFileUrl; // Assume the user provides the Dagshub file URL

    if (!dagshubFileUrl) {
        return res.render('index', { message: 'No training file URL provided' });
    }

    try {
        // Fetch the file from Dagshub
        const fileResponse = await axios.get(dagshubFileUrl, { responseType: 'stream' });

        // Prepare form data to send to Flask backend
        const form = new FormData();
        form.append('file', fileResponse.data, path.basename(dagshubFileUrl));

        // Send the file to Flask backend
        const flaskResponse = await axios.post('http://fraud_detection_app-backend-1:5010/upload_train_file', form, {
            headers: form.getHeaders()
        });

        const message = flaskResponse.data.message;
        res.render('index', { message });
    } catch (error) {
        console.error('Error fetching file from Dagshub or sending to Flask:', error);
        res.render('index', { message: 'Error processing training file' });
    }
});

// Retrain model (no changes needed)
app.post('/retrain_model', async (req, res) => {
    console.log("Start calling Model in Retrain Model");
    try {
        const response = await axios.post('http://fraud_detection_app-backend-1:5010/retrain');
        const messageRetrain = response.data.message;
        res.render('index', { messageRetrain });
    } catch (error) {
        console.error('Error calling Flask API:', error);
        res.render('index', { messageRetrain: 'Failed to retrain model' });
    }
});

app.listen(port, () => {
    console.log(`Frontend server running at http://localhost:${port}`);
});