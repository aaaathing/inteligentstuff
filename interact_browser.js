(async function(){
	// get video of current page
	let canvas = document.createElement("canvas")
	canvas.width=224
	canvas.height=224
	let ctx = canvas.getContext("2d", {willReadFrequently:true})
	let stream = await navigator.mediaDevices.getDisplayMedia({video:{displaySurface:"browser"},preferCurrentTab:true})
	let streamSettings = stream.getVideoTracks()[0].getSettings()
	let video = document.createElement("video")
	video.autoplay=true
	video.srcObject=stream
	
	let p = bot
	let prevHealth = p.health, prevFood = p.food

	let ws = new WebSocket("ws://localhost:8000");
	ws.onmessage = function(event) {
		let message = JSON.parse(event.data);
		let reward = 0
		if(prevHealth !== p.health){
			if(p.health < prevHealth) reward -= prevHealth-p.health
			prevHealth = p.health
		}
		if(p.food<4){
			reward -= (4-p.food)
		}
		if(prevFood !== p.food){
			if(p.food > prevFood) reward += prevFood-p.food
		}
		ws.send(JSON.stringify({reward}));
		let scale = Math.max(streamSettings.width/canvas.width, streamSettings.height/canvas.height)
		ctx.drawImage(video, 0,0, streamSettings.width/scale, streamSettings.height/scale)
		ws.send("video")
		ws.send(ctx.getImageData(0,0,canvas.width,canvas.height).data)
		ws.send("done")
	}
	ws.onclose = function(){
		for(let t of stream.getTracks()) t.stop()
	}

})()