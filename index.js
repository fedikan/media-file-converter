const mongoose = require('mongoose');
const fs = require('fs');
const axios = require('axios');
const { PutObjectCommand, S3Client, ObjectCannedACL } = require('@aws-sdk/client-s3');

const fileSchema = new mongoose.Schema({
  fileUrl: { type: String, required: true },
  originalUrl: { type: String, required: true },
  createdAt: { type: Date, default: Date.now },
  peaks: { type: Object, default: {} },
  meta: {
    fileType: { type: String },
    isDefault: { type: Boolean }
  },
  duration: {type: Number}, 
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
     // '_id': '6570adb624e2357fad3201fe'
    });

    for (const file of files) {
      try {

        const response = await axios.get(file.originalUrl, {
          responseType: 'arraybuffer'
        });

        if (response.status === 200) {
          const fileBuffer = Buffer.from(response.data);
          const analysisResponse = await convertAudioFile(fileBuffer)
          file.duration = analysisResponse.duration
          delete analysisResponse.duration
          file.peaks = analysisResponse
          await file.save()
          console.log(`${file.originalUrl} analyzed`);

        } else {
          console.error('Failed to download file:', file.originalUrl);
        }
      } catch (e) {
        console.log('Error processing file:', e)
      }
    }
    console.log("ITS DONe")
  } catch (error) {
    console.error('Error processing files:', error);
  }
}

async function uploadAudio(fileBuffer, { meta, userId }, outputFormat) {
  const timestamp = Math.round(Date.now() * Math.random());

  const peaks = await convertAudioFile(fileBuffer, outputFormat)


  return ({
    fileUrl: 'https://aiphoria-storage.ams3.cdn.digitaloceanspaces.com/' + optimizedFilename,
    meta: meta,
    userId,
  })
}

async function convertAudioFile(fileBuffer, outputFormat = 'wav') {
  // ... your convertAudioFile function here
  const formData = new FormData();
  const mimeType = `audio/${outputFormat}`;
  const blob = new Blob([fileBuffer], { type: mimeType });

  formData.append('file', blob, 'file.wav');

  const response = await axios.post('http://localhost:5000/track-meta', formData);

  if (response.status === 200) {
    return response.data;
  } else {
    throw new Error('Audio conversion failed');
  }
}

findAndProcessFiles();
