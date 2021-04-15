import React, { useState } from 'react';
import CloudConstants from '../constants/cloud.json';
import Form from './form.js';
import Dream from './dream.js';

function Root() {
  const [prompt, setPrompt] = useState('')
  const [urls, setUrls] = useState([])

  function fetchUrls() {
    const payload = {'jobDefinition': CloudConstants['pictionary']['jobDefinition'],
                     'jobQueue': CloudConstants['pictionary']['jobQueue'],
                     'bucket': CloudConstants['pictionary']['pictureBucket'],
                     'prompt': prompt}
    fetch(CloudConstants['pictionary']['apiEndpoint'],
    {
        method: 'POST',
        body: JSON.stringify(payload)
    })
    .then(response => response.text())
    .then(promise => setUrls(JSON.parse(promise)['urls']))
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    if (prompt) {
      fetchUrls()        
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