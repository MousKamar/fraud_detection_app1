const express = require('express');
const path = require('path');
const multer = require('multer');
const axios = require('axios');
const fs = require('fs');
const FormData = require('form-data');

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

// File upload setup
const upload = multer({ dest: 'uploads/' });


// Home Route (Landing page)
app.get('/', (req, res) => {
    res.render('index'); 
});


app.post('/upload', upload.single('file'),async (req, res) => {
    if (req.file) {
        const fileName = req.file.originalname;
        const filePath = req.file.path;
        const form = new FormData();
        form.append('file', fs.createReadStream(filePath), fileName);

       try {
            const response = await axios.post('http://localhost:5010/upload', form, {
                headers: form.getHeaders()
            });
            const predictions = response.data.predictions;
            console.log(predictions)
            res.render('result', { predictions });
            fs.unlinkSync(filePath);
        } catch (error) {
            console.error('Error sending file to Flask:', error);
            res.render('index');
        }
    } else {
        res.render('index');
    }
});


app.post('/upload_train_file',upload.single('train_file'), async (req, res) => {
    if (req.file) {
        const fileName = req.file.originalname;
        const filePath = req.file.path;
        const form = new FormData();
        form.append('file', fs.createReadStream(filePath), fileName);
       try {
            const response = await axios.post('http://localhost:5010/upload_train_file', form, {
                headers: form.getHeaders()
            });
            const message = response.data.message;
            res.render('index', { message });
            fs.unlinkSync(filePath);
        } catch (error) {
            console.error('Error sending file to Flask:', error);
            res.render('index');
        }
    } else {
        message = 'error fetching file';
        res.render('index',{message});
    }
});

app.post('/retrain_model', async (req, res) => {
    try {
        const response = await axios.post('http://localhost:5010/retrain');
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

