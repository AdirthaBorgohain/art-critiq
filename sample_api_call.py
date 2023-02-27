import requests

res = requests.post("http://0.0.0.0:8080/generate_review",
                    json={
                        "url": "https://openaccess-cdn.clevelandart.org/1946.83/1946.83_web.jpg",
                        "artist": "Edgar Degas (French, 1834–1917)",
                        "title": "Frieze of Dancers",
                        "type": "Painting",
                        "technique": "oil on fabric",
                        "review_type": "harsh"
                    })
print(res.json())
