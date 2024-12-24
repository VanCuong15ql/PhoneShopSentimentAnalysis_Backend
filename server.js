const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const app = express();

// Middleware
app.use(express.json());
app.use(cors());

// Connect to MongoDB
mongoose.connect('mongodb+srv://vancuongbui15ql:KMfuoqe6Gjn4UL8Z@cluster0.eglz7.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0', {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});

const userSchema = new mongoose.Schema({
  username: String,
  password: String,
});

const productSchema = new mongoose.Schema({
  name: String,
  image: String,
  comments: [
    {
      id: String,
      username: String,
      text: String,
      aspects: [String],
    },
  ],
});


const User = mongoose.model('User', userSchema);
const Product = mongoose.model('Product', productSchema);

// Routes
app.post('/register', async (req, res) => {
  const { username, password } = req.body;
  const user = new User({ username, password });
  await user.save();
  res.status(201).send('User registered');
});

app.post("/login", async (req, res) => {
  const { username, password } = req.body;
  const user = await User.findOne({ username, password });
  if (user) {
    res.status(200).json({ username: user.username });
  } else {
    res.status(401).send("Invalid credentials");
  }
});


app.get('/products', async (req, res) => {
  const products = await Product.find();
  res.status(200).json(products);
});

app.get('/products/:id', async (req, res) => {
  const product = await Product.findById(req.params.id);
  res.status(200).json(product);
});

// comment n aspect
// comment n aspect
const { spawn } = require('child_process');

app.post("/products/:id/comments", async (req, res) => {
  const { username, text } = req.body;
  const product = await Product.findById(req.params.id);
  if (product) {
    // Gọi mô hình Python để đánh giá các aspect
    const pythonProcess = spawn('python', ['predict_aspectsV2.py', text]);

    let aspects = [];
    let errorOccurred = false;

    pythonProcess.stdout.on('data', (data) => {
      aspects = data.toString().trim().split('\n');
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error(`stderr: ${data}`);
      errorOccurred = true;
    });

    pythonProcess.on('close', async (code) => {
      if (errorOccurred) {
        return res.status(500).send('Error evaluating aspects');
      }

      product.comments.push({ username, text, aspects });
      await product.save();
      res.status(201).send("Comment added with aspects");
    });
  } else {
    res.status(404).send("Product not found");
  }
});

app.put('/products/:productId/comments/:commentId', async (req, res) => {  
  const { productId, commentId } = req.params;  
  const { text } = req.body;  

  const product = await Product.findById(productId);  
  if (!product) return res.status(404).send('Product not found');  

  const comment = product.comments.id(commentId); // Tìm bình luận theo _id  
  if (!comment) return res.status(404).send('Comment not found');  

  comment.text = text; // Cập nhật nội dung bình luận  
  await product.save();  
  res.status(200).send('Comment updated');  
});




app.listen(5000, () => {
  console.log('Server is running on http://localhost:5000');
});