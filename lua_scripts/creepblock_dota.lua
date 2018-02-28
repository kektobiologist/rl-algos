dkjson = require( "game/dkjson" )
require 'utils.deepprint'


if CreepBlockAI == nil then
    _G.CreepBlockAI = class({}) 
end

function Activate()
    GameRules.CreepBlockAI = CreepBlockAI()
    GameRules.CreepBlockAI:InitGameMode()
end

function CreepBlockAI:InitGameMode()
    GameRules:SetShowcaseTime( 0 )
    GameRules:SetStrategyTime( 0 )
    GameRules:SetHeroSelectionTime( 0 )
    
    GameRules:GetGameModeEntity():SetCustomGameForceHero("npc_dota_hero_nevermore")
    
    ListenToGameEvent( "game_rules_state_change", Dynamic_Wrap( CreepBlockAI, 'OnGameRulesStateChange' ), self )
end

function CreepBlockAI:OnGameRulesStateChange()
    local s = GameRules:State_Get()  
    if  s == DOTA_GAMERULES_STATE_PRE_GAME then
        SendToServerConsole( "dota_all_vision 1" )
        SendToServerConsole( "dota_creeps_no_spawning  1" )
        SendToServerConsole( "dota_dev forcegamestart" )
        
    elseif  s == DOTA_GAMERULES_STATE_GAME_IN_PROGRESS then
        GameRules:GetGameModeEntity():SetThink("Setup", self, 5)
    end
end

-- STATE_GETMODEL = 0
-- STATE_SIMULATING = 1
-- STATE_SENDDATA = 2

STATE_SIMULATING = 0
STATE_RESETTING = 1

function CreepBlockAI:Setup()
    goodSpawn = Entities:FindByName( nil, "npc_dota_spawner_good_mid_staging" )
    goodWP = Entities:FindByName ( nil, "lane_mid_pathcorner_goodguys_1")
    heroSpawn = Entities:FindByName (nil, "dota_goodguys_tower2_mid"):GetAbsOrigin() + Vector(-200, -200, 0)
    hero = Entities:FindByName (nil, "npc_dota_hero_nevermore")
    t1 =  Entities:FindByName(nil, "dota_goodguys_tower1_mid")
    t1Pos = t1:GetAbsOrigin()
    t1_c = t1Pos.y + t1Pos.x + 2000

    -- for calculating distance from midlane
    t3_rad = Entities:FindByName(nil, 'dota_goodguys_tower3_mid'):GetAbsOrigin()
    t3_dire = Entities:FindByName(nil, 'dota_badguys_tower3_mid'):GetAbsOrigin()
    
    -- NOTE: removing camera centering
    -- PlayerResource:SetCameraTarget(0, hero)
    
    heroSpeed = hero:GetBaseMoveSpeed()
    -- had weights here

    ai_state = STATE_RESETTING
    ep = -1 -- will become zero upon Start()
    -- self:Reset()
    GameRules:GetGameModeEntity():SetThink("MainLoop", self, 1) -- wait a few seonds before starting stuff
    hero:SetContextThink("BotThink", function() return self:BotLoop() end, 0.2)
end

--------------------------------------------------------------------------------


function CreepBlockAI:MainLoop()
    local startPoint = Vector(goodSpawn:GetAbsOrigin().x, goodSpawn:GetAbsOrigin().y, 0)
    local endPoint = Vector(goodWP:GetAbsOrigin().x, goodWP:GetAbsOrigin().y, 0)
    print(startPoint)
    print(endPoint)
    DebugDrawLine(Vector(0,0,0), endPoint, 255, 0, 255, true, 1)

    if ai_state == STATE_RESETTING then
        self:Reset()
        self:Start()
        ai_state = STATE_SIMULATING
    elseif ai_state == STATE_SIMULATING then
        -- don't really need to do shit?
    else
        Warning('some shit has gone bad')
    end
    -- if ai_state == STATE_GETMODEL then                           
    --  Say(hero, "Starting Episode " .. ep, false)
        
    --  self:Start()  
    --  ai_state = STATE_SIMULATING 
    -- elseif ai_state == STATE_SIMULATING then
    
    -- elseif ai_state == STATE_SENDDATA then
    --  Say(hero, "Episode Ended", false)
    --  ai_state = STATE_GETMODEL
    -- else
    --  Warning("Some shit has gone bad..")
    -- end
    
    return 1
end

function CreepBlockAI:BotLoop()
    if ai_state ~= STATE_SIMULATING then
        return 0.2
    end
            
    self:UpdatePositions()
    
    -- skip if callback is still pending
    if isComputing then
        return 0.2
    end
    -- local terminal, action= self:UpdateSAR()
    self:UpdateSAR(
        function(terminal, action)

            if terminal then
                -- self:Reset()
                ai_state = STATE_RESETTING
                return
            end
            
            hero:MoveToPosition(hPos + action)
                    
            t = t + 1
    
            return 
        end)

    -- trying to make sure packets aren't skipped...
    -- nope
    return 0.2
    -- if terminal then
    --     self:Reset()
    --     ai_state = STATE_SENDDATA
    --     return 0.2
    -- end
    
    -- hero:MoveToPosition(hPos + action)
            
    -- t = t + 1
    
    -- return 0.2
end

--------------------------------------------------------------------------------

function CreepBlockAI:UpdatePositions()
    hPos = hero:GetAbsOrigin()
    cPos = {}
    for i = 1,4 do
        cPos[i] = creeps[i]:GetAbsOrigin()
    end
end

-- cb expected to be called with prevTerminal, action
function CreepBlockAI:UpdateSAR(cb)   
    local s_t = {}
    -- fuck this
    -- for i = 1,4 do
    --     s_t[i*2-1] = (cPos[i].x - hPos.x) / heroSpeed
    --     s_t[i*2] = (cPos[i].y - hPos.y) / heroSpeed
    -- end
    -- use coordinates of everything as state
    -- manually normalize coordinates ...
    for i = 1, 4 do
        s_t[i*2-1] = (cPos[i].x +2500)/2500.0
        s_t[i*2] = (cPos[i].y + 2250)/2250.0
    end
    s_t[9] = (hPos.x + 2500)/2500.0
    s_t[10] = (hPos.y + 2250)/2250.0
    -- add normalized hero angle as 2 variables, sin theta and cos theta
    -- print(hero:GetAngles()[2])
    local angle  = (2.0 * math.pi *  hero:GetAngles()[2]) / 360.0
    s_t[11] = math.sin(angle)
    s_t[12] = math.cos(angle)
    local prevReward = 0
    if t > 0 then
        local reward = 0
        for i = 1,4 do
            local dist = (cPos[i] - last_cPos[i]):Length2D()
            local hdist = (hPos - cPos[i]):Length2D()
            local hdistLast = (hPos - last_cPos[i]):Length2D()
            if hdist < 500 then
                if dist < 20 then
                    reward = reward + 0.5
                elseif dist < 40 then
                    reward = reward + 0.35  
                elseif dist < 60 then
                    reward = reward + 0.2
                else
                -- just give a little reward for being close...
                -- no, should give reward for getting CLOSER to creeps?
                    -- reward = reward + 0.005
                end
                -- if hdist < hdistLast then
                --     -- got closer to creep
                --     reward = reward + 0.005
                -- end
            end
        end
        
        -- SAR[t-1]['r'] = reward
        prevReward = reward
    end
    
    last_cPos = cPos
    
    -- local terminal = false
    local prevTerminal = false
    local c1 = hPos.y + hPos.x + 100
    local min_dist = 100000
    local target = Vector(0,0,0)
    for i = 1,4 do
        local c2 = cPos[i].y - cPos[i].x
        local x = (c1 - c2) / 2.0
        local y = -x + c1
        if cPos[i].y > (-cPos[i].x + c1) then
            -- SAR[t-1]['r'] = SAR[t-1]['r'] - 1
            -- prevReward = prevReward - 1
            -- just make it uniformly -1
            reward = -1
            -- terminal = true
            prevTerminal = true
        end
        if cPos[i].y > (-cPos[i].x + t1_c) then
            -- terminal = true
            prevTerminal = true
        end
    end

    -- check if hero is too far from lane
    -- local distFromLane = math.abs(hPos.x - hPos.y)/math.sqrt(2)
    local distFromLane = CalcDistanceToLineSegment2D(hPos, t3_rad, t3_dire)
    -- print(distFromLane)
    -- print(hPos)
    if distFromLane > 400 then
        prevReward = -10
        prevTerminal = true
    end

    
    -- if not terminal then
    --     SAR[t] = { s=s_t, a={action.x, action.y}, r=0 }
    -- end

    -- send prevTerminal and prevReward along with the state to run
    
    -- local action = self:Run(s_t)
    local action = self:RunAsync(s_t, prevReward, prevTerminal,
        function(action)
            cb(prevTerminal, action)
        end)
    -- return  prevTerminal, action
end

--------------------------------------------------------------------------------
-- cb expected to be called with reply, err
function CreepBlockAI:SendPostRequest(json, cb)
    local req = CreateHTTPRequestScriptVM('POST', 'http://127.0.0.1:5000/' )
    req:SetHTTPRequestRawPostBody("application/json", json)
    req:Send( function( result )
        for k,v in pairs( result ) do
            if k == "Body" then
                local jsonReply, pos, err = dkjson.decode(v, 1, nil)
                if err then
                    print("JSON Decode Error: ", err)
                    print("Sent Message: ", json)
                    print("Msg Body: ", v)
                    cb(nil)
                else
                    -- print('sent message: ' .. json)
                    -- print( tostring(jsonReply) )
                    cb(jsonReply)
                    -- DeepPrintTable(jsonReply)
                    -- packet:ProcessPacket(jsonReply.Type, jsonReply)
                    
                    -- if jsonReply.Type == packet.TYPE_AUTH then
                    --     webserverFound = true
                    --     print("Connected Successfully to Backend Server")
                    -- elseif jsonReply.Type == packet.TYPE_POLL then
                    --     print("Received Update from Server")
                    -- end
                end
                --break
            end
        end
    end )
end

-- cb expected to be called with action
function CreepBlockAI:RunAsync(state, prevReward, prevTerminal, cb)
    local dat = {}
    dat['state'] = state
    dat['prevReward'] = prevReward
    dat['prevTerminal'] = prevTerminal
    dat['t'] = t
    dat['ep'] = ep
    -- print(dat['state'])
    isComputing = true
    self:SendPostRequest(dkjson.encode(dat), function(reply)
            -- default to 0, 0 action
            local action = Vector(0, 0, 0)
            if reply ~= nil then 
                action = Vector(reply['action']['x'], reply['action']['y'], 0)
            end
            DebugDrawCircle(hPos + action, Vector(0,255,0), 255, 25, false, 0.2)
            isComputing = false
            cb(action)
        end)

end

function CreepBlockAI:Reset()   
    hero:Stop()
    SendToServerConsole( "dota_dev hero_refresh" )
    FindClearSpaceForUnit(hero, heroSpawn + Vector(150,-150,0), true)
    
    if creeps ~= nil then
        for i = 1,4 do
            creeps[i]:ForceKill(false)
        end 
    end
end

function CreepBlockAI:Start()
    t = 0
    -- SAR = {}
    -- SAR['ep'] = ep
    ep = ep + 1
    -- use variable to keep track if callback is pending
    isComputing = false
    creeps = {}
    -- NOTE: REMOVING RANDOMNESS
    for i=1,3 do
        creeps[i] = CreateUnitByName( "npc_dota_creep_goodguys_melee" , goodSpawn:GetAbsOrigin() + RandomVector( RandomFloat( 0, 0 ) ), true, nil, nil, DOTA_TEAM_GOODGUYS )              
    end
    creeps[4] = CreateUnitByName( "npc_dota_creep_goodguys_ranged" , goodSpawn:GetAbsOrigin() + RandomVector( RandomFloat( 0, 0 ) ), true, nil, nil, DOTA_TEAM_GOODGUYS )
    
    for i = 1,4 do 
        creeps[i]:SetInitialGoalEntity( goodWP )
    end     
end    

--------------------------------------------------------------------------------

-- function CreepBlockAI:Run(s_t)
--     local action = Vector(0,0,0)
--     local fc1 = RELU(FC(s_t, self.W1, self.b1))
--     local fc2 = RELU(FC(fc1, self.W2, self.b2))
--     local fc3 = FC(fc2, self.W3, self.b3)
    
--     local weight = {}
--     local max_i = 1
--     for i = 1,20 do
--         weight[i] = math.exp(fc3[i])
--         if weight[i] > weight[max_i] then
--             max_i = i
--         end
--     end
            
--     for i = 1,20 do
--         if i == max_i then
--             DebugDrawCircle(hPos + Vector(fc3[20+i],fc3[40+i],0), Vector(0,255,0), 255, 25, true, 0.2)
--         else
--             DebugDrawCircle(hPos + Vector(fc3[20+i],fc3[40+i],0), Vector(255,0,0), 255*weight[i]/weight[max_i], 25, true, 0.2)
--         end
--     end
--     action = Vector(fc3[max_i+20],fc3[max_i+40],0)
    
--     return action
-- end

-- function FC(x, W, b)
--     local y = {}
--     for j = 1,#b do
--         y[j] = 0
--         for i = 1,#x do
--             y[j] = y[j] + x[i]*W[i][j]
--         end
--         y[j] = y[j] + b[j]
--     end
--     return y
-- end

-- function RELU(x)
--     local y = {}
--     for i = 1,#x do
--         if x[i] < 0 then
--             y[i] = 0
--         else
--             y[i] = x[i]
--         end
--     end
--     return y
-- end
