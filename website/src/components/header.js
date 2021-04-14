import React from 'react';

export default ({title}) => (
  <div className='header'>
    <div className="grid" style={{gridTemplateColumns: "repeat(5, 1fr)"}}>
      <div></div>
      <div></div>
      <div>
        {title}
      </div>
      <div></div>
      <div></div>
    </div>
  </div>
);