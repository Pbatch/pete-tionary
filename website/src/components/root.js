import React, { useState } from 'react';
import CloudConstants from '../constants/cloud.json';
import { v4 } from 'node-uuid';


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

  const handleSubmit = (event) => {
    event.preventDefault()
    fetchUrls()
  }

  const images = urls.map((url) => {
    return (
      <img key={v4()} src={url} alt={url} style={{border: "1px solid black"}}/>
      )
    });

  return (
    <>
      <div>
        {images}
      </div>
      <form onSubmit={handleSubmit}>
        <input type="text" onChange={e => setPrompt(e.target.value)} />
        <button type="submit">Submit prompt</button>
      </form>
      
    </>
  )
}

export default Root;