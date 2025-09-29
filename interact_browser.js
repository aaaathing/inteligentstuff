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
	
	let p
	let prevHealth, prevFood
	let prevMouseDown = [false, false]

	let ws = new WebSocket("ws://localhost:8000");
	ws.onmessage = function(event) {
		let msg = JSON.parse(event.data);
		let reward = 0

		if(p !== window.bot){
			p = window.bot
			// if bot has just been defined
			if(p) prevHealth = p.health, prevFood = p.food
		}
		// p and bot are the same
		if(p){
			if(msg.action[0] || msg.action[1]){
				document.dispatchEvent(new MouseEvent('mousemove', {
					movementX: (msg.action[0]-msg.action[1])*0.1,
					movementY: (msg.action[2]-msg.action[3])*0.1,
				}))
			}

			bot.setControlState('forward', msg.action[4]>0.5)
			bot.setControlState('back', msg.action[5]>0.5)
			bot.setControlState('left', msg.action[6]>0.5)
			bot.setControlState('right', msg.action[7]>0.5)
			bot.setControlState('jump', msg.action[8]>0.5)
			bot.setControlState('sneak', msg.action[9]>0.5)
			bot.setControlState('sprint', msg.action[10]>0.5)

			let mouseDown = msg.action[11]>0.5
			if(prevMouseDown[0] !== mouseDown){
				prevMouseDown[0] = mouseDown
				if(mouseDown[0]) document.dispatchEvent(new MouseEvent('mousedown', {button:0}))
				else document.dispatchEvent(new MouseEvent('mouseup', {button:0}))
			}
			mouseDown = msg.action[12]>0.5
			if(prevMouseDown[1] !== mouseDown){
				prevMouseDown[1] = mouseDown
				if(mouseDown[1]) document.dispatchEvent(new MouseEvent('mousedown', {button:2}))
				else document.dispatchEvent(new MouseEvent('mouseup', {button:2}))
			}

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