import React, { useState } from 'react';
import CloudConstants from '../constants/cloud.json';
import Form from './form.js';
import Dream from './dream.js';

function Root() {
  const [prompt, setPrompt] = useState('')
  const [urls, setUrls] = useState([])

  function submitBatchJob() {
    const payload = {'jobDefinition': CloudConstants['pictionary']['jobDefinition'],
                     'jobQueue': CloudConstants['pictionary']['jobQueue'],
                     'bucket': CloudConstants['pictionary']['pictureBucket'],
                     'prompt': prompt}
    fetch(CloudConstants['pictionary']['apiEndpoint'], {
      method: 'POST',
      body: JSON.stringify(payload)
    })
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    if (prompt) {
      // submitBatchJob()    
      const newUrls = Array(3)
      const region = CloudConstants["pictionary"]["region"]
      const bucket = CloudConstants["pictionary"]["pictureBucket"]
      var i;
      for (i=0; i<3; i++) {
        newUrls[i] = `https://s3-${region}.amazonaws.com/${bucket}/prompt=${prompt}-seed=${i}.jpg`
      }  
      setUrls(newUrls)
      console.log(urls)
    } 
  }

  return (
    <div id='root'>
      <Dream urls={urls} />
      <Form handleSubmit={handleSubmit} setPrompt={setPrompt} />
    </div>
  )
}

export default Root;