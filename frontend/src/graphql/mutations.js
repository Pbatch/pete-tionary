import gql from 'graphql-tag'

export const CreateMessage = gql`
  mutation($url: String!, $username: String!, $round: Int!, $roomName: String!) {
    createMessage(input: {
      roomName: $roomName
      round: $round
      url: $url
      username: $username
    }) {
      id 
      roomName
      round
      url
      username
      ttl
    }
  }
`

export const CreateRoom = gql`
  mutation($roomName: String!) {
    createRoom(roomName: $roomName) 
    {
      roomName
      ttl
    }
  }
`

export const DeleteRoom = gql`
  mutation($roomName: String!) {
    deleteRoom(roomName: $roomName) 
    {
      roomName
    }
  }
`

export const DeleteMessage = gql`
  mutation($id: String!) {
    deleteMessage(id: $id) 
    {
      id 
    }
  }
`