const mongoose = require('mongoose');
const axios = require('axios');
const { PutObjectCommand, S3Client, ObjectCannedACL } = require('@aws-sdk/client-s3');

const fileSchema = new mongoose.Schema({
  fileUrl: { type: String, required: true },
  originalUrl: { type: String, required: true },
  createdAt: { type: Date, default: Date.now },
  meta: {
    fileType: { type: String },
    isDefault: { type: Boolean }
  },
  userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User' }
});


const File = mongoose.model('File', fileSchema);

// Assuming you have a connection to MongoDB and the File model is imported

async function findAndProcessFiles() {

  try {
    await mongoose.connect('mongodb://threeheadcrow:125373nogolem@104.248.247.171:8888/openai_completions?authMechanism=DEFAULT&authSource=admin');
    console.log('mongo connected')
    const files = await File.find({
      'meta.fileType': { $in: ['musicgen', 'bark', 'riffusion'] },
      'fileUrl': /\.wav$/
    });

    for (const file of files) {
      console.log(file.fileUrl)
      const response = await axios.get(file.fileUrl, {
        responseType: 'arraybuffer'
      });

      if (response.status === 200) {
        const fileBuffer = Buffer.from(response.data);
        const optimizedFile = await uploadAudio(fileBuffer, { meta: file.meta, userId: file.userId }, 'wav');
        file.fileUrl = optimizedFile.fileUrl
        await file.save()
        console.log('File processed:', optimizedFile);
      } else {
        console.error('Failed to download file:', file.fileUrl);
      }
    }
  } catch (error) {
    console.error('Error processing files:', error);
  }
}
const uploadToS3 = async (buffer, filename) => {
  const s3Client = new S3Client({
    endpoint: 'https://ams3.digitaloceanspaces.com',
    forcePathStyle: false,
    region: 'us-east-1',
    credentials: {
      accessKeyId: 'DO004C3EYEC2JLN9FJ6X',
      secretAccessKey: 'rqSKvnbR7uIEDW6XrAxhaNHpUPEF1dcUIqzEWYsN8Tw',
    },

  });
  const params = {
    Bucket: 'aiphoria-storage',
    Key: filename,
    Body: buffer,
    ACL: 'public-read'
  };

  try {
    await s3Client.send(new PutObjectCommand(params));
    console.log('Successfully uploaded file:', filename);
  } catch (err) {
    console.log('Error uploading file:', err);
    throw new Error('Error uploading file to S3');
  }
}

async function uploadAudio(fileBuffer, { meta, userId }, outputFormat) {
  const timestamp = Math.round(Date.now() * Math.random());
  const optimizedFilename = `optimized_${timestamp}.mp3`;

  const convertedBuffer = await convertAudioFile(fileBuffer, outputFormat)

  await uploadToS3(convertedBuffer, optimizedFilename);

  return ({
    fileUrl: 'https://aiphoria-storage.ams3.cdn.digitaloceanspaces.com/' + optimizedFilename,
    meta: meta,
    userId,
  })
}

async function convertAudioFile(fileBuffer, outputFormat) {
  // ... your convertAudioFile function here
  const formData = new FormData();
  const mimeType = `audio/${outputFormat}`;
  const blob = new Blob([fileBuffer], { type: mimeType });

  formData.append('file', blob, 'file.wav');

  const response = await axios.post('http://localhost:5000/convert', formData, {
    responseType: 'arraybuffer',
  });

  if (response.status === 200) {
    return response.data;
  } else {
    throw new Error('Audio conversion failed');
  }
}

findAndProcessFiles();