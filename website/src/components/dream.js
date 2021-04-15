import { v4 } from 'node-uuid';

function Dream({urls}) {
  const images = urls.map((url) => {
    return (
      <img key={v4()} src={url} alt={url} />
      )
    });

  return (
    <div id='imageContainer' className='grid' style={{gridTemplateColumns: "repeat(5, 1fr)"}}>
      {images}
    </div>
  );
}

export default Dream;